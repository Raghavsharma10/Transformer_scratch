def execute(self, command):
        """Execute a DDE command."""
        pData = c_char_p(command)
        cbData = DWORD(len(command) + 1)
        hDdeData = DDE.ClientTransaction(pData, cbData, self._hConv, HSZ(), CF_TEXT, XTYP_EXECUTE, TIMEOUT_ASYNC, LPDWORD())
        if not hDdeData:
            raise DDEError("Unable to send command", self._idInst)
        DDE.FreeDataHandle(hDdeData)