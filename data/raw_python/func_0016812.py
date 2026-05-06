def decode(self, descriptor):
        """ Produce a list of dictionaries for each dimension in this transcoder """
        i = iter(descriptor)
        n = len(self._schema)

        # Add the name key to our schema
        schema = self._schema + ('name',)
        # For each dimensions, generator takes n items off iterator
        # wrapping the descriptor, making a tuple with the dimension
        # name appended
        tuple_gen = (tuple(itertools.islice(i, n)) + (d, )
            for d in self._dimensions)

        # Generate dictionary by mapping schema keys to generated tuples
        return [{ k: v for k, v in zip(schema, t) } for t in tuple_gen]