def add_options(self, parser):
        """Add command-line options for this plugin.

        The base plugin class adds --with-$name by default, used to enable the
        plugin. 
        """
        parser.add_option("--with-%s" % self.name,
                          action="store_true",
                          dest=self.enable_opt,
                          help="Enable plugin %s: %s" %
                          (self.__class__.__name__, self.help())
                          )