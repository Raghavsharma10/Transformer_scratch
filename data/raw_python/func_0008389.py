def _read_data(self, dataset, format=None):
        """Reads a data file and returns and iterable that can be used as
        testing or training data."""
        # Attempt to detect file format if "format" isn't specified
        if not format:
            format_class = formats.detect(dataset)
        else:
            if format not in formats.AVAILABLE.keys():
                raise ValueError("'{0}' format not supported.".format(format))
            format_class = formats.AVAILABLE[format]
        return format_class(dataset).to_iterable()