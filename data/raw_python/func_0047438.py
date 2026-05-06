def set_reference(self,ref):
    """Set the reference sequence

    :param ref: reference sequence
    :type ref: string

    """
    self._options = self._options._replace(reference = ref)