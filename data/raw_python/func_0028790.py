def advise(self, item, stop=False):
        """Request updates when DDE data changes."""
        hszItem = DDE.CreateStringHandle(self._idInst, item, CP_WINUNICODE)
        hDdeData = DDE.ClientTransaction(LPBYTE(), 0, self._hConv, hszItem, CF_TEXT, XTYP_ADVSTOP if stop else XTYP_ADVSTART, TIMEOUT_ASYNC, LPDWORD())
        DDE.FreeStringHandle(self._idInst, hszItem)
        if not hDdeData:
            raise DDEError("Unable to %s advise" % ("stop" if stop else "start"), self._idInst)
        DDE.FreeDataHandle(hDdeData)