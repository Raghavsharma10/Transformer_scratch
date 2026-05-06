def create_objective_bank_hierarchy(self, alias, desc, genus):
        """
        Create a bank hierarchy with the given alias
        :param alias:
        :return:
        """
        url_path = self._urls.hierarchy()
        data = {
            'id': re.sub(r'[ ]', '', alias.lower()),
            'displayName': {
                'text': alias
            },
            'description': {
                'text': desc
            },
            'genusTypeId': str(genus)
        }
        return self._post_request(url_path, data)