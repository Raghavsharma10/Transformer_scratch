def get_page(self, page_id):
        """ Get short page info and body html code """
        try:
            result = self._request('/getpage/',
                                   {'pageid': page_id})
            return TildaPage(**result)
        except NetworkError:
            return []