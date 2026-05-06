def get_page_full(self, page_id):
        """ Get full page info and full html code """
        try:
            result = self._request('/getpagefull/',
                                   {'pageid': page_id})
            return TildaPage(**result)
        except NetworkError:
            return []