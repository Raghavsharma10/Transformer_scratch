def change_password(self, current_password, new_password):
        """
        Changes account password.
        @param current_password: current password
        @param new_password: new password
        """
        attrs = {sconstant.A_BY: sconstant.V_NAME}
        account = SOAPpy.Types.stringType(data=self.auth_token.account_name,
                                          attrs=attrs)

        params = {sconstant.E_ACCOUNT: account,
                  sconstant.E_OLD_PASSWORD: current_password,
                  sconstant.E_PASSWORD: new_password}

        self.invoke(zconstant.NS_ZIMBRA_ACC_URL,
                    sconstant.ChangePasswordRequest,
                    params)