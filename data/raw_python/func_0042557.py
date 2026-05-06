def get_page_export(self, page_id):
        """ Get short page info for export and body html code """
        try:
            result = self._request('/getpageexport/',
                                   {'pageid': page_id})
            return TildaPage(**result)
        except NetworkError:
            return []