def zLoadFile(self, fileName, append=None):
        """Loads a zmx file into the DDE server"""
        reply = None
        if append:
            cmd = "LoadFile,{},{}".format(fileName, append)
        else:
            cmd = "LoadFile,{}".format(fileName)
        reply = self._sendDDEcommand(cmd)
        if reply:
            return int(reply) #Note: Zemax returns -999 if update fails.
        else:
            return -998