def zPushLens(self, update=None, timeout=None):
        """Copy lens in the Zemax DDE server into LDE"""
        reply = None
        if update == 1:
            reply = self._sendDDEcommand('PushLens,1', timeout)
        elif update == 0 or update is None:
            reply = self._sendDDEcommand('PushLens,0', timeout)
        else:
            raise ValueError('Invalid value for flag')
        if reply:
            return int(reply)   # Note: Zemax returns -999 if push lens fails
        else:
            return -998