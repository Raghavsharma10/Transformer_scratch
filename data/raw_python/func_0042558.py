def get_page_full_export(self, page_id):
        """ Get full page info for export and body html code """
        try:
            result = self._request('/getpagefullexport/',
                                   {'pageid': page_id})
            return TildaPage(**result)
        except NetworkError:
            return []