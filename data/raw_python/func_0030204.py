def page(self):
        '''Current page.'''
        page = self.request.GET.get(self.page_param)
        if not page:
            return 1
        try:
            page = int(page)
        except ValueError:
            self.invalid_page()
            return 1
        if page<1:
            self.invalid_page()
            return 1
        return page