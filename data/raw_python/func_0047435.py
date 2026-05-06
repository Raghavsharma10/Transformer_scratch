def set_query_sequence(self,seq):
    """Assign the query sequence.
    
    :param seq: sequence of the query
    :type seq: string

    """
    self._options = self._options.replace(query_sequence = seq)