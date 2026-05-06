def pieces(array, chunk_size):
        """Yield successive chunks from array/list/string.
        Final chunk may be truncated if array is not evenly divisible by chunk_size."""
        for i in range(0, len(array), chunk_size): yield array[i:i+chunk_size]