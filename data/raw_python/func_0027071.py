def get_settings(self, link):
        """
        URL of service settings
        """
        return reverse(
            'servicesettings-detail', kwargs={'uuid': link.service.settings.uuid}, request=self.context['request'])