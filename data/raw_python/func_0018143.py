def get_admin_urls_for_registration(self):
        """
        Utilised by Wagtail's 'register_admin_urls' hook to register urls for
        used by any associated ModelAdmin instances
        """
        urls = []
        for instance in self.modeladmin_instances:
            urls.extend(instance.get_admin_urls_for_registration())
        return urls