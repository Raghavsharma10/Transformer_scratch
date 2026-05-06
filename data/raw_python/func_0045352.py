def get_from_cache(key):
        """
        Args:
            key (str):
        Return:
            (dict): from json string
        """

        try:
            value = cache.get(key)
            return json.loads(value), key if value else None
        except Exception as e:
            # todo should add log.error()
            return None