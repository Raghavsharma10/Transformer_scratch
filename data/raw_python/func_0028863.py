def get_urls(self):
        """
        Add aditional moderate url.
        """
        from django.conf.urls import url
        urls = super(AdminModeratorMixin, self).get_urls()
        info = self.model._meta.app_label, self.model._meta.model_name
        return [
            url(r'^(.+)/moderate/$',
                self.admin_site.admin_view(self.moderate_view),
                name='%s_%s_moderate' % info),
        ] + urls