def create(self):
        """
        Create the plugin page in all languages and fill dummy content.
        """
        plugin = CMSPlugin.objects.filter(plugin_type=self.apphook)
        if plugin.exists():
            log.debug('Plugin page for "%s" plugin already exist, ok.',
                      self.apphook)
            raise plugin

        page, created = super(CmsPluginPageCreator, self).create()

        if created:
            # Add a plugin with content in all languages to the created page.
            # But only on new created page
            for placeholder_slot in self.placeholder_slots:
                self.fill_content(page, placeholder_slot)

        return page, created