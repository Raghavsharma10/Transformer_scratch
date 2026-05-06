def create(self):
        """
        creates an empty configuration file
        """
        if not self.exists():
            # create new empyt config file based on template
            self.config.add_section("lametric")
            self.config.set("lametric", "client_id", "")
            self.config.set("lametric", "client_secret", "")

            # save new config
            self.save()

            # stop here, so user can set his config
            sys.exit(
                "An empty config file '{}' has been created. Please set "
                "the corresponding LaMetric API credentials.".format(
                    self._filename
                )
            )