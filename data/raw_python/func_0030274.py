def get_home_page(self):
        """
        Return the published home page.
        Used for 'parent' in cms.api.create_page()
        """
        try:
            home_page_draft = Page.objects.get(
                is_home=True, publisher_is_draft=True)
        except Page.DoesNotExist:
            log.error('ERROR: "home page" doesn\'t exists!')
            raise RuntimeError('no home page')
        return home_page_draft