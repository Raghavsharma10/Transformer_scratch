def contribute_static(self):
        """Contributes static and media file serving settings to an existing section."""

        options = self.options
        if options['compile'] or not options['use_static_handler']:
            return

        from django.core.management import call_command

        settings = self.settings
        statics = self.section.statics
        statics.register_static_map(settings.STATIC_URL, settings.STATIC_ROOT)
        statics.register_static_map(settings.MEDIA_URL, settings.MEDIA_ROOT)

        call_command('collectstatic', clear=True, interactive=False)