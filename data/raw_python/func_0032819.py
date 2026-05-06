def routeMessage(self, sender, target, value, messageID):
        """
        Implement L{IMessageRouter.routeMessage} by synchronously locating an
        account via L{axiom.userbase.LoginSystem.accountByAddress}, and
        delivering a message to it by calling a method on it.
        """
        router = self._routerForAccount(target)
        if router is not None:
            router.routeMessage(sender, target, value, messageID)
        else:
            reverseRouter = self._routerForAccount(sender)
            reverseRouter.routeAnswer(sender, target,
                                      Value(DELIVERY_ERROR, ERROR_NO_USER),
                                      messageID)