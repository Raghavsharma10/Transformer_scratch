def static_url(redis, path):
    """Gets the static path for a file"""
    file_hash = get_cache_buster(redis, path)
    return "%s/%s?v=%s" % (oz.settings["static_host"], path, file_hash)