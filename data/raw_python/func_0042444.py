def get_urls(self):
        """Add our dashboard view to the admin urlconf. Deleted the default index."""
        from django.conf.urls import patterns, url
        from views import DashboardWelcomeView

        urls = super(AdminMixin, self).get_urls()
        del urls[0]
        custom_url = patterns(
            '',
            url(r'^$', self.admin_view(DashboardWelcomeView.as_view()), name="index")
        )

        return custom_url + urls