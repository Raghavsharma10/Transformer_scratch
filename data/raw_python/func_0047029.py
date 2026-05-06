def original_query_sequence_length(self):
    """Similar to get_get_query_sequence_length, but it also includes
    hard clipped bases
    if there is no cigar, then default to trying the sequence

    :return: the length of the query before any clipping
    :rtype: int
    """
    if not self.is_aligned() or not self.entries.cigar:
      return self.query_sequence_length # take the naive approach
    # we are here with something aligned so take more intelligent cigar apporach
    return sum([x[0] for x in self.cigar_array if re.match('[HMIS=X]',x[1])])