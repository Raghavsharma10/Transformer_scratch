def make_batch(size, graph):
        """
        Split graphs into n sized chunks.
        See: http://stackoverflow.com/a/1915307/758157

        :param size: int
        :param graph: graph
        :return: graph
        """
        i = iter(graph)
        chunk = list(islice(i, size))
        while chunk:
            yield chunk
            chunk = list(islice(i, size))