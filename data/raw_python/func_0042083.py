def apply_config_file(self, filename):
        """
            Add options from config file to self.values
            Leave alone existing values that are not an instance of Default
        """
        def extractor(template, options):
            """Ignore things that are existing non default values"""
            for name, val in options:
                normalised = self.normalise_key(name)
                if normalised in self.values and not isinstance(self.values[normalised], Default):
                    continue
                else:
                    yield name, val

        items = json.load(open(filename)).items()
        self.use_options(items, extractor)