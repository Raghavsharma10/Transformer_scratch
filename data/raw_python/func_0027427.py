def setup(self):
        """Setup filter (only called when filter is actually used)."""
        super(RequireJSFilter, self).setup()

        excluded_files = []
        for bundle in self.excluded_bundles:
            excluded_files.extend(
                map(lambda f: os.path.splitext(f)[0],
                    bundle.contents)
            )

        if excluded_files:
            self.argv.append(
                'exclude={0}'.format(','.join(excluded_files))
            )