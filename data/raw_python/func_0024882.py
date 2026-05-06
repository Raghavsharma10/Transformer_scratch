def _get_config_path(self):
        """
        Return a sensible configuration path for caching config
        settings.
        """
        org = self.service.space.org.name
        space = self.service.space.name
        name = self.name

        return "~/.predix/%s/%s/%s.json" % (org, space, name)