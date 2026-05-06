def sequence(self):
    """A strcutre is defined so get,
    if the sequence is not already there, get the sequence from the reference

    Always is returned on the positive strand for the MappingGeneric

    :param ref_dict: reference dictionary (only necessary if sequence has not been set already)
    :type ref_dict: dict()
    """
    if not self._options.ref:
      raise ValueError("ERROR: sequence is not defined and reference is undefined")
    #chr = self.range.chr
    return self.get_sequence(self._options.ref)