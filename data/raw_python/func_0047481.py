def set_payload(self,val):
    """Set a payload for this object

    :param val: payload to be stored
    :type val: Anything that can be put in a list
    """
    self._options = self._options._replace(payload = val)