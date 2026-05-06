def configure(self, options, conf):
        """Configure the plugin and system, based on selected options.

        The base plugin class sets the plugin to enabled if the enable option
        for the plugin (self.enable_opt) is true.
        """
        self.conf = conf
        if hasattr(options, self.enable_opt):
            self.enabled = getattr(options, self.enable_opt)