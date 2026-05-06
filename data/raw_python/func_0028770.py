def zSaveFile(self, fileName):
        """Saves the lens currently loaded in the server to a Zemax file """
        cmd = "SaveFile,{}".format(fileName)
        reply = self._sendDDEcommand(cmd)
        return int(float(reply.rstrip()))