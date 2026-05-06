def get_values(self, *args, **kwargs):
        """
        Convenience method that for simple single tag queries will
        return just the values to be iterated on.
        """
        if isinstance(args[0], list):
            raise ValueError("Can only get_values() for a single tag.")

        response = self.get_datapoints(*args, **kwargs)
        for value in response['tags'][0]['results'][0]['values']:
            yield [datetime.datetime.utcfromtimestamp(value[0]/1000),
                   value[1],
                   value[2]]