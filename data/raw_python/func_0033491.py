def get_cache_buster(redis, path):
    """Gets the cache buster value for a given file path"""
    return escape.to_unicode(redis.hget("cache-buster:{}:v3".format(oz.settings["s3_bucket"]), path))