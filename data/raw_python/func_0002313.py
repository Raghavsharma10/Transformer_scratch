def get_urls(self):
        """
        Introduce more urls
        """
        urls = super(PageAdmin, self).get_urls()
        my_urls = [
            url(r'^get_layout/$', self.admin_site.admin_view(self.get_layout_view))
        ]
        return my_urls + urls