def request(self, item, timeout=5000):
        """Request data from DDE service."""
        hszItem = DDE.CreateStringHandle(self._idInst, item, CP_WINUNICODE)
        #hDdeData = DDE.ClientTransaction(LPBYTE(), 0, self._hConv, hszItem, CF_TEXT, XTYP_REQUEST, timeout, LPDWORD())
        pdwResult = DWORD(0)
        hDdeData = DDE.ClientTransaction(LPBYTE(), 0, self._hConv, hszItem, CF_TEXT, XTYP_REQUEST, timeout, byref(pdwResult))
        DDE.FreeStringHandle(self._idInst, hszItem)
        if not hDdeData:
            raise DDEError("Unable to request item", self._idInst)

        if timeout != TIMEOUT_ASYNC:
            pdwSize = DWORD(0)
            pData = DDE.AccessData(hDdeData, byref(pdwSize))
            if not pData:
                DDE.FreeDataHandle(hDdeData)
                raise DDEError("Unable to access data in request function", self._idInst)
            DDE.UnaccessData(hDdeData)
        else:
            pData = None
        DDE.FreeDataHandle(hDdeData)
        return pData