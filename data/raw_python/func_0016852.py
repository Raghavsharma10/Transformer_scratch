def _feed_impl(self, cube, data_sources, data_sinks, global_iter_args):
        """ Implementation of staging_area feeding """
        session = self._tf_session
        FD = self._tf_feed_data
        LSA = FD.local

        # Get source strides out before the local sizes are modified during
        # the source loops below
        src_types = LSA.sources.keys()
        src_strides = [int(i) for i in cube.dim_extent_size(*src_types)]
        src_staging_areas = [[LSA.sources[t][s] for t in src_types]
            for s in range(self._nr_of_shards)]

        compute_feed_dict = { ph: cube.dim_global_size(n) for
            n, ph in FD.src_ph_vars.iteritems() }
        compute_feed_dict.update({ ph: getattr(cube, n) for
            n, ph in FD.property_ph_vars.iteritems() })

        chunks_fed = 0

        which_shard = itertools.cycle([self._shard(d,s)
            for s in range(self._shards_per_device)
            for d, dev in enumerate(self._devices)])

        while True:
            try:
                # Get the descriptor describing a portion of the RIME
                result = session.run(LSA.descriptor.get_op)
                descriptor = result['descriptor']
            except tf.errors.OutOfRangeError as e:
                montblanc.log.exception("Descriptor reading exception")

            # Quit if EOF
            if descriptor[0] == -1:
                break

            # Make it read-only so we can hash the contents
            descriptor.flags.writeable = False

            # Find indices of the emptiest staging_areas and, by implication
            # the shard with the least work assigned to it
            emptiest_staging_areas = np.argsort(self._inputs_waiting.get())
            shard = emptiest_staging_areas[0]
            shard = which_shard.next()

            feed_f = self._feed_executors[shard].submit(self._feed_actual,
                data_sources.copy(), cube.copy(),
                descriptor, shard,
                src_types, src_strides, src_staging_areas[shard],
                global_iter_args)

            compute_f = self._compute_executors[shard].submit(self._compute,
                compute_feed_dict, shard)

            consume_f = self._consumer_executor.submit(self._consume,
                data_sinks.copy(), cube.copy(), global_iter_args)

            self._inputs_waiting.increment(shard)

            yield (feed_f, compute_f, consume_f)

            chunks_fed += 1

        montblanc.log.info("Done feeding {n} chunks.".format(n=chunks_fed))