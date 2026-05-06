def do_search(self, string):
        """Search Ndrive for filenames containing the given string."""
        results = self.n.doSearch(string, full_path = self.current_path)

        if results:
            for r in results:
                self.stdout.write("%s\n" % r['path'])