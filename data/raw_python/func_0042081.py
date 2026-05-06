def use_options(self, options, extractor=None):
        """
            If extractor isn't specified, then just update self.values with options.

            Otherwise update values with whatever the result of calling extractor with
            our template and these options returns

            Also make sure all keys are transformed into valid python attribute names
        """
        # Extract if necessary
        if not extractor:
            extracted = options
        else:
            extracted = extractor(self.template, options)

        # Get values as [(key, val), ...]
        if isinstance(extracted, dict):
            extracted = extracted.items()

        # Add our values if there are any
        # Normalising the keys as we go along
        if extracted is not None:
            for key, val in extracted:
                self.values[self.normalise_key(key)] = val