def get_info(self, account, params={}):
        """
        Gets account info.
        @param account: account to get info for
        @param params: parameters to retrieve
        @return: AccountInfo
        """
        res = self.invoke(zconstant.NS_ZIMBRA_ADMIN_URL,
                          sconstant.GetInfoRequest,
                          params)

        return res