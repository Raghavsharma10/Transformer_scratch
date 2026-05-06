def show_entry_points(self):
        """
        Show entry points for a module

        @returns: 0 for success or 1 if error

        """
        found = False
        for entry_point in \
                pkg_resources.iter_entry_points(self.options.show_entry_points):
            found = True
            try:
                plugin = entry_point.load()
                print(plugin.__module__)
                print("   %s" % entry_point)
                if plugin.__doc__:
                    print(plugin.__doc__)
                print
            except ImportError:
                pass
        if not found:
            self.logger.error("No entry points found for %s" \
                    % self.options.show_entry_points)
            return 1
        return 0