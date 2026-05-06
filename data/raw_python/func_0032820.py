def routeAnswer(self, originalSender, originalTarget, value, messageID):
        """
        Implement L{IMessageRouter.routeMessage} by synchronously locating an
        account via L{axiom.userbase.LoginSystem.accountByAddress}, and
        delivering a response to it by calling a method on it and returning a
        deferred containing its answer.
        """
        router = self._routerForAccount(originalSender)
        return router.routeAnswer(originalSender, originalTarget,
                                  value, messageID)