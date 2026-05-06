def emit(self):
    """Get a mapping from a transcript

    :return: One random Transcript sequence
    :rtype: sequence
    """
    i = self.options.rand.get_weighted_random_index(self._weights)
    return self._transcriptome.transcripts[i]