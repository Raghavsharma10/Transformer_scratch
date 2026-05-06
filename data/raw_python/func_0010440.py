def parse(self):
        """
        Convert ACIS 'll' value into separate latitude and longitude.
        """
        super(AcisIO, self).parse()

        # This is more of a "mapping" step than a "parsing" step, but mappers
        # only allow one-to-one mapping from input fields to output fields.
        for row in self.data:
            if 'meta' in row:
                row = row['meta']
            if 'll' in row:
                row['longitude'], row['latitude'] = row['ll']
                del row['ll']