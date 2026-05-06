def _init_update_po_files(self, domains):
        """Update or initialize the `.po` translation files"""
        for language in settings.TRANSLATIONS:
            for domain, options in domains.items():
                if language == options['default']: continue  # Default language of the domain doesn't need translations
                if os.path.isfile(_po_path(language, domain)):
                    # If the translation already exists, update it, keeping the parts already translated
                    self._update_po_file(language, domain, options['pot'])
                else:
                    # The translation doesn't exist, create a new translation file
                    self._init_po_file(language, domain, options['pot'])