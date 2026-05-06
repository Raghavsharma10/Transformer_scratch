def doAction(self, loginMethod, actionClass):
        """
        Show the form for the requested action.
        """
        loginAccount = loginMethod.account
        return actionClass(
            self,
            loginMethod.localpart + u'@' + loginMethod.domain,
            loginAccount)