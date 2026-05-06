def copy_image_from_url(url, cache_dir=None, use_cache=True):
    """ Copy image from given URL and return upload metadata.
    """
    return cache_image_data(cache_dir, hashlib.sha1(url).hexdigest(), ImgurUploader().upload, url, use_cache=use_cache)