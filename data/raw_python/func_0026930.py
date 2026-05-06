def version(self, event):
        """
        Shows version information.
        """
        name = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        return "%s [%s]" % (settings.GNOTTY_VERSION_STRING, name)