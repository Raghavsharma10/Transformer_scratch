def biggus_chunk(chunk_key, biggus_array, masked):
        """
        A function that lazily evaluates a biggus.Chunk. This is useful for
        passing through as a dask task so that we don't have to compute the
        chunk in order to compute the graph.

        """
        if masked:
            array = biggus_array.masked_array()
        else:
            array = biggus_array.ndarray()

        return biggus._init.Chunk(chunk_key, array)