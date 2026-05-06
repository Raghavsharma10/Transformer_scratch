def Log(self, messages):
    """
    Parameters:
     - messages
    """
    self._seqid += 1
    d = self._reqs[self._seqid] = defer.Deferred()
    self.send_Log(messages)
    return d