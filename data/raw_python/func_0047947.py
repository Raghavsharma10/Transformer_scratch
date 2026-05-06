def get_genus_type(self):
        """Gets the genus type of this object.

        return: (osid.type.Type) - the genus type of this object
        compliance: mandatory - This method must be implemented.

        """
        if self._my_genus_type_map is None:
            url_path = '/handcar/services/learning/types/' + self._my_map['genusTypeId']
#            url_str = self._base_url + '/types/' + self._my_map['genusTypeId']
#            self._my_genus_type_map = self._load_json(url_str)
            self._my_genus_type_map = self._get_request(url_path)
        return Type(self._my_genus_type_map)