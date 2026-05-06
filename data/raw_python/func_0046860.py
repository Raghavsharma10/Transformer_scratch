def delete_objective_bank_hierarchy(self, alias):
        """
        Delete this bank hierarchy
        :param alias:
        :return:
        """
        url_path = self._urls.hierarchy(alias)
        return self._delete_request(url_path)