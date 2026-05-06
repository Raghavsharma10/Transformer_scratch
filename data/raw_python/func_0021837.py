def show_pypi_releases(self):
        """
        Show PyPI releases for the last number of `hours`

        @returns: 0 = success or 1 if failed to retrieve from XML-RPC server

        """
        try:
            hours = int(self.options.show_pypi_releases)
        except ValueError:
            self.logger.error("ERROR: You must supply an integer.")
            return 1
        try:
            latest_releases = self.pypi.updated_releases(hours)
        except XMLRPCFault as err_msg:
            self.logger.error(err_msg)
            self.logger.error("ERROR: Couldn't retrieve latest releases.")
            return 1

        for release in latest_releases:
            print("%s %s" % (release[0], release[1]))
        return 0