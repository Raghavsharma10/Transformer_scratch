def set_cache_buster(redis, path, hash):
    """Sets the cache buster value for a given file path"""
    redis.hset("cache-buster:{}:v3".format(oz.settings["s3_bucket"]), path, hash)