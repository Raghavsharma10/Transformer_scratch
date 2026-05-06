def set_to_cache(vk):
        """
        Args:
            vk (tuple): obj data (dict), obj key(str)

        Return:
            tuple: value (dict), key (string)
        """
        v, k = vk

        try:
            cache.set(k, json.dumps(v), settings.CACHE_EXPIRE_DURATION)
        except Exception as e:
            pass
            # todo should add log.error()
        return v, k