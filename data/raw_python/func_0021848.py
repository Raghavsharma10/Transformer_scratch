def show_entry_map(self):
        """
        Show entry map for a package

        @param dist: package
        @param type: srting

        @returns: 0 for success or 1 if error
        """
        pprinter = pprint.PrettyPrinter()
        try:
            entry_map = pkg_resources.get_entry_map(self.options.show_entry_map)
            if entry_map:
                pprinter.pprint(entry_map)
        except pkg_resources.DistributionNotFound:
            self.logger.error("Distribution not found: %s" \
                    % self.options.show_entry_map)
            return 1
        return 0