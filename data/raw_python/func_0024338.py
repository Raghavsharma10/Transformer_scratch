def get_all(self, cat):
        """
        if data can't found in cache then it will be fetched from db,
         parsed and stored to cache for each lang_code.

        :param cat: cat of catalog data
        :return:
        """
        return self._get_from_local_cache(cat) or self._get_from_cache(cat) or self._get_from_db(cat)