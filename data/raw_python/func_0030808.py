def sendBoxCommand(self, command, box, requiresAnswer=True):
        """
        Send a command across the wire with the given C{juice.Box}.

        Returns a Deferred which fires with the response C{juice.Box} when it
        is received, or fails with a C{juice.RemoteJuiceError} if an error is
        received.

        If the Deferred fails and the error is not handled by the caller of
        this method, the failure will be logged and the connection dropped.
        """
        if self._outstandingRequests is None:
            return fail(CONNECTION_LOST)
        box[COMMAND] = command
        tag = self._nextTag()
        if requiresAnswer:
            box[ASK] = tag
            result = self._outstandingRequests[tag] = Deferred()
        else:
            result = None
        box.sendTo(self)
        return result