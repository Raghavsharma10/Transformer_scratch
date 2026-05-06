def _make_nodes(self, dsk_graph, array, iteration_order, masked,
                    top=False):
        """
        Recursive function that returns the dask items for the given array.

        NOTE: Currently assuming that all tasks are a tuple, with the second
        item being the keys used to index the source of the respective input
        array.

        """
        cache_key = _array_id(array, iteration_order, masked)
        # By the end of this function Nodes will be a dictionary with one item
        # per chunk to be processed for this array.
        nodes = self._node_cache.get(cache_key, None)

        if nodes is None:
            if hasattr(array, 'streams_handler'):
                nodes = self._make_stream_handler_nodes(dsk_graph, array,
                                                        iteration_order,
                                                        masked)
            else:
                nodes = {}
                chunks = []

                name = '{}\n{}'.format(array.__class__.__name__, array.shape)
                biggus_chunk_func = self.lazy_chunk_creator(name)

                chunk_index_gen = biggus._init.ProducerNode.chunk_index_gen
                for chunk_key in chunk_index_gen(array.shape,
                                                 iteration_order[::-1]):
                    biggus_array = array[chunk_key]
                    pretty_key = ', '.join(map(slice_repr, chunk_key))
                    chunk_id = ('chunk shape: {}\nsource key: [{}]\n\n{}'
                                ''.format(biggus_array.shape, pretty_key,
                                          uuid.uuid4()))
                    task = (biggus_chunk_func, chunk_key, biggus_array, masked)
                    chunks.append(task)
                    assert chunk_id not in dsk_graph
                    dsk_graph[chunk_id] = task
                    nodes[chunk_id] = task
            self._node_cache[cache_key] = nodes
        return nodes