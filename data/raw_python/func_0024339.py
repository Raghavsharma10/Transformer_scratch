def _fill_get_item_cache(self, catalog, key):
        """
        get from redis, cache locally then return

        :param catalog: catalog name
        :param key:
        :return:
        """
        lang = self._get_lang()
        keylist = self.get_all(catalog)
        self.ITEM_CACHE[lang][catalog] = dict([(i['value'],  i['name']) for i in keylist])
        return self.ITEM_CACHE[lang][catalog].get(key)