def get_info(self, params={}):
        """
        Gets mailbox info.
        @param params: params to retrieve
        @return: AccountInfo
        """
        res = self.invoke(zconstant.NS_ZIMBRA_ACC_URL,
                          sconstant.GetInfoRequest,
                          params)

        return res