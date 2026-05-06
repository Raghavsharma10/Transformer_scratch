def _deliverAnswer(self, answer):
        """
        Attempt to deliver an answer to a message sent to this store, via my
        store's parent's L{IMessageRouter} powerup.

        @param answer: an L{AlreadyAnswered} that contains an answer to a
        message sent to this store.
        """
        router = self.siteRouter
        if answer.deliveryDeferred is None:
            d = answer.deliveryDeferred = router.routeAnswer(
                answer.originalSender, answer.originalTarget, answer.value,
                answer.messageID)
            def destroyAnswer(result):
                answer.deleteFromStore()
            def transportErrorCheck(f):
                answer.deliveryDeferred = None
                f.trap(MessageTransportError)
            d.addCallbacks(destroyAnswer, transportErrorCheck)
            d.addErrback(log.err)