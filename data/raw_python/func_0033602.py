def set_cache_buster(self, path, hash):
        """Sets the cache buster value for a given file path"""
        oz.aws_cdn.set_cache_buster(self.redis(), path, hash)