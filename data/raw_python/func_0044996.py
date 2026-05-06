def links(self):
        """
        returns {attr1: href, attr2: href2}
        """
        dlinks = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict) and value['link']:
                dlinks[key] = value['link']
        return dlinks