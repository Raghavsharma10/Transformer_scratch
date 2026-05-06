def show_pypi_changelog(self):
        """
        Show detailed PyPI ChangeLog for the last `hours`

        @returns: 0 = sucess or 1 if failed to retrieve from XML-RPC server

        """
        hours = self.options.show_pypi_changelog
        if not hours.isdigit():
            self.logger.error("Error: You must supply an integer.")
            return 1

        try:
            changelog = self.pypi.changelog(int(hours))
        except XMLRPCFault as err_msg:
            self.logger.error(err_msg)
            self.logger.error("ERROR: Couldn't retrieve changelog.")
            return 1

        last_pkg = ''
        for entry in changelog:
            pkg = entry[0]
            if pkg != last_pkg:
                print("%s %s\n\t%s" % (entry[0], entry[1], entry[3]))
                last_pkg = pkg
            else:
                print("\t%s" % entry[3])

        return 0