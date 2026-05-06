def decensor(post_info: dict, site_url: str = DEFAULT_SITE) -> dict:
    "Decensor a post info dict from Danbooru API if needed."
    return post_info \
           if "md5" in post_info else fill_missing_info(post_info, site_url)