def process(self):
        """Run the analysis across all files found in the given paths.

        Each file is loaded once and all plugins are run against it before
        loading the next file.

        """
        for filename in self.hairball_files(self.paths, self.extensions):
            if not self.options.quiet:
                print(filename)
            try:
                if self.cache:
                    scratch = self.cache.load(filename)
                else:
                    scratch = kurt.Project.load(filename)
            except Exception:  # pylint: disable=W0703
                traceback.print_exc()
                continue
            for plugin in self.plugins:
                # pylint: disable=W0212
                plugin._process(scratch, filename=filename)