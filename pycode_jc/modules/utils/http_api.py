import traceback
import requests
import logging


def json_data_post(payload, post_url, files=None):
    try:
        response = requests.post(post_url, json=payload, files=files, timeout=5)
        logging.info("Post data success. to [%s]. data: [%s]. files: [%s].Return: [%s]. Status code: [%s]",
                     post_url, payload, files, response.content, response.status_code)
        return response
    except:
        logging.exception("Post files and data failed. to [%s]. data: [%s]. Error: [%s]", post_url, payload,
                          traceback.format_exc())
        return None
