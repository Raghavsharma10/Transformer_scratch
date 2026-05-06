def fill_missing_info(info: dict, site_url: str = DEFAULT_SITE) -> dict:
    "Add missing info in a censored post info dict."
    try:
        md5, ext = find_censored_md5ext(info["id"])
    except TypeError:  # None returned by find_..
        return info

    sample_ext = "jpg" if ext != "zip" else "webm"

    if info["id"] > 2_800_000:
        site_url   = site_url.rstrip("/")
        file_url   = f"{site_url}/data/{md5}.{ext}"
        sample_url = f"{site_url}/data/sample/sample-{md5}.{sample_ext}"
    else:
        server     = "raikou2" if info["id"] > 850_000 else "raikou1"
        url_base   = f"https://{server}.donmai.us"
        file_url   = f"{url_base}/{md5[:2]}/{md5[2:4]}/{md5}.{ext}"
        sample_url = (f"{url_base}/sample/{md5[:2]}/{md5[2:4]}/"
                      f"sample-{md5}.{sample_ext}")

    if info["image_width"] < 850:
        sample_url = file_url

    return {**info, **{
        "file_ext":         ext,
        "md5":              md5,
        "file_url":         file_url,
        "large_file_url":   sample_url,
        "preview_file_url": (f"https://raikou4.donmai.us/preview/"
                             f"{md5[:2]}/{md5[2:4]}/{md5}.jpg"),
    }}