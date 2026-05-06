def page_url(self, page):
        '''
        Returns URL for page, page is included as query parameter.

        Can be redefined by keyword argument
        '''
        if page is not None and page != 1:
            return self.url.qs_set(**{self.page_param: page})
        elif page is not None:
            return self.url.qs_delete('page')