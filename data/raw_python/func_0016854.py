def _consume_impl(self, data_sinks, cube, global_iter_args):
        """ Consume """

        LSA = self._tf_feed_data.local
        output = self._tfrun(LSA.output.get_op)

        # Expect the descriptor in the first tuple position
        assert len(output) > 0
        assert LSA.output.fed_arrays[0] == 'descriptor'

        descriptor = output['descriptor']
        # Make it read-only so we can hash the contents
        descriptor.flags.writeable = False

        dims = self._transcoder.decode(descriptor)
        cube.update_dimensions(dims)

        # Obtain and remove input data from the source cache
        try:
            input_data = self._source_cache.pop(descriptor.data)
        except KeyError:
            raise ValueError("No input data cache available "
                "in source cache for descriptor {}!"
                    .format(descriptor))

        # For each array in our output, call the associated data sink
        gen = ((n, a) for n, a in output.iteritems() if not n == 'descriptor')

        for n, a in gen:
            sink_context = SinkContext(n, cube,
                self.config(), global_iter_args,
                cube.array(n) if n in cube.arrays() else {},
                a, input_data)

            _supply_data(data_sinks[n], sink_context)