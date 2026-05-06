def _make_stream_handler_nodes(self, dsk_graph, array, iteration_order,
                                   masked):
        """
        Produce task graph entries for an array that comes from a biggus
        StreamsHandler.

        This is essentially every type of array that isn't already a thing on
        disk/in-memory. StreamsHandler arrays include all aggregations and
        elementwise operations.

        """
        nodes = {}
        handler = array.streams_handler(masked)
        input_iteration_order = handler.input_iteration_order(iteration_order)

        def input_keys_transform(input_array, keys):
            if hasattr(input_array, 'streams_handler'):
                handler = input_array.streams_handler(masked)
                # Get the transformer of the input array, and apply it to the
                # keys.
                input_transformer = getattr(handler,
                                            'output_keys', None)
                if input_transformer is not None:
                    keys = input_transformer(keys)
            return keys

        sources_keys = []
        sources_chunks = []
        for input_array in array.sources:
            # Bring together all chunks that influence the same part of this
            # (resultant) array.
            source_chunks_by_key = {}
            sources_chunks.append(source_chunks_by_key)
            source_keys = []
            sources_keys.append(source_keys)

            # Make nodes for the source arrays (if they don't already exist)
            # before we do anything else.
            input_nodes = self._make_nodes(dsk_graph, input_array,
                                           input_iteration_order, masked)

            for chunk_id, task in input_nodes.items():
                chunk_keys = task[1]
                t_keys = chunk_keys
                t_keys = input_keys_transform(array, t_keys)
                source_keys.append(t_keys)
                this_key = str(t_keys)
                source_chunks_by_key.setdefault(this_key,
                                                []).append([chunk_id, task])

        sources_keys_grouped = key_grouper.group_keys(array.shape,
                                                      *sources_keys)
        for slice_group, sources_keys_group in sources_keys_grouped.items():
            # Each group is entirely independent and can have its own task
            # without knowledge of results from items in other groups.

            t_keys = tuple(slice(*slice_tuple) for slice_tuple in slice_group)

            all_chunks = []
            for source_keys, source_chunks_by_key in zip(sources_keys_group,
                                                         sources_chunks):
                dependencies = tuple(
                        the_id
                        for keys in source_keys
                        for the_id, task in source_chunks_by_key[str(keys)])
                # Uniquify source_keys, but keep the order.
                dependencies = tuple(_unique_everseen(dependencies))

                def normalize_keys(keys, shape):
                    result = []
                    for key, dim_length in zip(keys, shape):
                        result.append(key_grouper.normalize_slice(key,
                                                                  dim_length))
                    return tuple(result)

                # If we don't have the same chunks for all inputs then we
                # should combine them before passing them on to the handler.
                # TODO: Fix slice equality to deal with 0 and None etc.
                if not all(t_keys == normalize_keys(keys, array.shape)
                           for keys in source_keys):
                    combined = self.collect(array[t_keys], masked, chunk=True)
                    new_task = (combined, ) + dependencies
                    new_id = ('chunk shape: {}\n\n{}'
                              ''.format(array[t_keys].shape, uuid.uuid()))
                    dsk_graph[new_id] = new_task
                    dependencies = (new_id, )

                all_chunks.append(dependencies)

            pivoted = all_chunks

            sub_array = array[t_keys]
            handler = sub_array.streams_handler(masked)
            name = getattr(handler, 'nice_name', handler.__class__.__name__)

            if hasattr(handler, 'axis'):
                name += '\n(axis={})'.format(handler.axis)
            # For ElementwiseStreams handlers, use the function that they wrap
            # (e.g "add")
            if hasattr(handler, 'operator'):
                name = handler.operator.__name__

            n_sources = len(array.sources)
            handler_of_chunks_fn = self.create_chunks_handler_fn(handler,
                                                                 n_sources,
                                                                 name)

            shape = sub_array.shape
            if all(key == slice(None) for key in t_keys):
                subset = ''
            else:
                pretty_index = ', '.join(map(slice_repr, t_keys))
                subset = 'target subset [{}]\n'.format(pretty_index)

            # Flatten out the pivot so that dask can dereferences the IDs
            source_chunks = [item for sublist in pivoted for item in sublist]
            task = tuple([handler_of_chunks_fn, t_keys] + source_chunks)
            shape_repr = ', '.join(map(str, shape))
            chunk_id = 'chunk shape: ({})\n\n{}{}'.format(shape_repr,
                                                          subset,
                                                          uuid.uuid4())
            assert chunk_id not in dsk_graph
            dsk_graph[chunk_id] = task
            nodes[chunk_id] = task
        return nodes