import glob
import os
import time
import uuid
from urllib.error import HTTPError
from urllib.request import urlretrieve
import boto3
import requests
import yaml
from pyunsplash import PyUnsplash
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from utils.logger import logger
from utils.misc import (
    create_bucket_if_not_exists,
    upload_dir_to_s3,
)


MAX_NUM_REQUEST_PER_HOUR = 50
NUM_IMAGES_PER_PAGE = 60
FASHION_STYLES = [
    "americancasual",
    "casual",
    "chic",
    "dandy",
    "formal",
    "girlish",
    "golf",
    "retro",
    "romantic",
    "sports",
    "street",
    "gorpcore",
]

if __name__ == "__main__":
    with open(os.path.join("config", "config.yaml"), encoding="utf-8") as file_path:
        config = yaml.safe_load(file_path)

    api_key = config["environment"]["unsplash_api_key"]
    profile_name = config["environment"]["iam_profile_name"]
    region_name = config["environment"]["region_name"]
    images_dir = config["environment"]["ebs_images_dir"]
    bucket = config["environment"]["s3_bucket"]
    base_prefix = config["environment"]["s3_base_prefix"]
    dataset_prefix = config["environment"]["s3_dataset_prefix"]

    source = config["data"]["source"]
    num_images = config["data"]["num_images"]
    query = config["data"]["query"]
    is_street_snap = config["data"]["is_street_snap"]

    boto_session = boto3.Session(profile_name=profile_name, region_name=region_name)

    os.makedirs(images_dir, exist_ok=True)

    # Crawling images from Unsplash or Musinsa

    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    logger.info(
        "There are %d images in total before downloading.",
        len(image_paths),
    )

    tot_count = 0

    if source is None or source.lower() == "unsplash":
        unsplash = PyUnsplash(api_key=api_key)

        quotient, remainder = divmod(num_images, MAX_NUM_REQUEST_PER_HOUR)

        for i in range(quotient + 1):
            count = MAX_NUM_REQUEST_PER_HOUR if i < quotient else remainder
            if count > 0:
                photos = unsplash.photos(
                    type_="random",
                    count=count,
                    featured=True,
                    query=query,
                )

                for j, photo in enumerate(photos.entries):
                    start = time.time()

                    photo_id = photo.id
                    response = requests.get(photo.link_download, allow_redirects=True)
                    image_name = f"image_{source}_{photo_id}.png"
                    with open(os.path.join(images_dir, image_name), "wb") as file:
                        file.write(response.content)

                    tot_count += 1
                    msg = (
                        f"The '{tot_count}' image, '{image_name}' has been downloaded."
                    )
                    logger.info(msg)

                    end = time.time()
                    time.sleep(
                        3600 // MAX_NUM_REQUEST_PER_HOUR - round(end - start) + 1
                    )

    elif source.lower() == "musinsa":
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        fashion_styles = [None] if is_street_snap else FASHION_STYLES
        quotient, remainder = divmod(num_images, NUM_IMAGES_PER_PAGE)

        for fashion_style in fashion_styles:
            photo_tag = "" if fashion_style is None else f"{fashion_style}_"

            for i in range(quotient + 1):
                url = (
                    f"https://magazine.musinsa.com/index.php?m=street&ordw=inc&_mon=&p={i + 1}#listStart"
                    if is_street_snap
                    else f"https://www.musinsa.com/app/styles/lists?use_yn_360=&style_type={fashion_style}&brand=&model=&tag_no=&max_rt=&min_rt=&display_cnt={NUM_IMAGES_PER_PAGE}&list_kind=big&sort=date&page={i + 1}"
                )
                driver.get(url)

                count = NUM_IMAGES_PER_PAGE if i < quotient else remainder
                if count > 0:
                    for j in range(count):
                        if is_street_snap:
                            driver.find_elements(
                                by=By.CSS_SELECTOR, value=".articleImg"
                            )[j].click()
                            image_url = driver.find_elements(
                                by=By.CSS_SELECTOR, value=".lbox"
                            )[0].get_attribute("href")

                        else:
                            driver.find_elements(
                                by=By.CSS_SELECTOR, value=".style-list-item__thumbnail"
                            )[j].click()
                            image_url = (
                                driver.find_element(
                                    by=By.CSS_SELECTOR, value=".detail_slider"
                                )
                                .find_elements(by=By.TAG_NAME, value="img")[1]
                                .get_attribute("src")
                            )

                        try:
                            photo_id = str(uuid.uuid4()).split("-", maxsplit=1)[0]

                            image_name = f"image_{source}_{photo_tag}{photo_id}.png"
                            urlretrieve(
                                image_url,
                                os.path.join(
                                    images_dir,
                                    image_name,
                                ),
                            )

                            tot_count += 1
                            msg = f"The '{tot_count}' image, '{image_name}' has been downloaded."
                            logger.info(msg)

                        except HTTPError:
                            msg_suffix = (
                                ""
                                if fashion_style is None
                                else f" of the {fashion_style} fashion style"
                            )
                            msg = f"There was an error downloading the '{j + 1}' image on page '{i + 1}'{msg_suffix}."
                            logger.warning(msg)

                        driver.get(url)

    else:
        raise ValueError(
            "Currently, the only supported crawl target sources are 'unsplash' or 'musinsa'."
        )

    # Uploading images to S3

    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    logger.info(
        "There are %d images in total after downloading.",
        len(image_paths),
    )

    bucket = (
        create_bucket_if_not_exists(boto_session, region_name, logger=logger)
        if bucket is None
        else bucket
    )

    upload_dir_to_s3(
        boto_session,
        images_dir,
        bucket,
        f"{base_prefix}/{dataset_prefix}/images",
        file_ext_to_excl=["DS_Store"],
        logger=logger,
    )

    logger.info("Crawling and uploading images is complete.")
