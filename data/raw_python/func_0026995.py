def _get_cache_key(self, args, kwargs):
        """ Returns key to be used in cache """
        hash_input = json.dumps({'name': self.name, 'args': args, 'kwargs': kwargs}, sort_keys=True)
        # md5 is used for internal caching, not need to care about security
        return hashlib.md5(hash_input).hexdigest()