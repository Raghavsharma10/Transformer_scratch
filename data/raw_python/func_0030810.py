def sendPacket(self, completeBox):
        """
        Send a juice.Box to my peer.

        Note: transport.write is never called outside of this method.
        """
        assert not self.__locked, "You cannot send juice packets when a connection is locked"
        if self._startingTLSBuffer is not None:
            self._startingTLSBuffer.append(completeBox)
        else:
            if debug:
                log.msg("Juice send: %s" % pprint.pformat(dict(completeBox.iteritems())))

            self.transport.write(completeBox.serialize())