def _callback(self, wType, uFmt, hConv, hsz1, hsz2, hDdeData, dwData1, dwData2):
        """DdeCallback callback function for processing Dynamic Data Exchange (DDE)
        transactions sent by DDEML in response to DDE events

        Parameters
        ----------
        wType    : transaction type (UINT)
        uFmt     : clipboard data format (UINT)
        hConv    : handle to conversation (HCONV)
        hsz1     : handle to string (HSZ)
        hsz2     : handle to string (HSZ)
        hDDedata : handle to global memory object (HDDEDATA)
        dwData1  : transaction-specific data (DWORD)
        dwData2  : transaction-specific data (DWORD)

        Returns
        -------
        ret      : specific to the type of transaction (HDDEDATA)
        """
        if wType == XTYP_ADVDATA:  # value of the data item has changed [hsz1 = topic; hsz2 = item; hDdeData = data]
            dwSize = DWORD(0)
            pData = DDE.AccessData(hDdeData, byref(dwSize))
            if pData:
                item = create_string_buffer('\000' * 128)
                DDE.QueryString(self._idInst, hsz2, item, 128, CP_WINANSI)
                self.callback(pData, item.value)
                DDE.UnaccessData(hDdeData)
                return DDE_FACK
            else:
                print("Error: AccessData returned NULL! (err = %s)"% (hex(DDE.GetLastError(self._idInst))))
        if wType == XTYP_DISCONNECT:
            print("Disconnect notification received from server")

        return 0