def post_process(self, settings):
        """
        Perform post processing methods on settings according to their
        definition in manifest.

        Post process methods are implemented in their own method that have the
        same signature:

        * Get arguments: Current settings, item name and item value;
        * Return item value possibly patched;

        Args:
            settings (dict): Loaded settings.

        Returns:
            dict: Settings object possibly modified (depending from applied
                post processing).

        """
        for k in settings:
            # Search for post process rules for setting in manifest
            if k in self.settings_manifesto and \
               self.settings_manifesto[k].get('postprocess', None) is not None:
                rules = self.settings_manifesto[k]['postprocess']

                # Chain post process rules from each setting
                for method_name in rules:
                    settings[k] = getattr(self, method_name)(
                        settings,
                        k,
                        settings[k]
                    )

        return settings