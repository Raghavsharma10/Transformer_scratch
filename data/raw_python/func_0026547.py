def _read_config(self):
        """Read this component's configuration from the database"""

        try:
            self.config = self.componentmodel.find_one(
                {'name': self.uniquename})
        except ServerSelectionTimeoutError:  # pragma: no cover
            self.log("No database access! Check if mongodb is running "
                     "correctly.", lvl=critical)
        if self.config:
            self.log("Configuration read.", lvl=verbose)
        else:
            self.log("No configuration found.", lvl=warn)