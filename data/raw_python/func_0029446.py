def _get_xpath_for(self, prop):
        """ :return: the configured xpath for a given property """

        xpath = self._data_map.get(prop)
        return getattr(xpath, 'xpath', xpath)