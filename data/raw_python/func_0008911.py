def _get_chunk_edges(self, NN, chunk_size, chunk_overlap):
        """
        Given the size of the array, calculate and array that gives the
        edges of chunks of nominal size, with specified overlap
        Parameters
        ----------
        NN : int
            Size of array
        chunk_size : int
            Nominal size of chunks (chunk_size < NN)
        chunk_overlap : int
            Number of pixels chunks will overlap
        Returns
        -------
        start_id : array
            The starting id of a chunk. start_id[i] gives the starting id of
            the i'th chunk
        end_id : array
            The ending id of a chunk. end_id[i] gives the ending id of
            the i'th chunk
        """
        left_edge = np.arange(0, NN - chunk_overlap, chunk_size)
        left_edge[1:] -= chunk_overlap
        right_edge = np.arange(0, NN - chunk_overlap, chunk_size)
        right_edge[:-1] = right_edge[1:] + chunk_overlap
        right_edge[-1] = NN
        right_edge = np.minimum(right_edge, NN)
        return left_edge, right_edge