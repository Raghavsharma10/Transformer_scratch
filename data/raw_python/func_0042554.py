def get_pages_list(self, project_id):
        """ Get pages list """
        try:
            result = self._request('/getpageslist/',
                                   {'projectid': project_id})
            return [TildaPage(**p) for p in result]
        except NetworkError:
            return []