def routeAnswer(self, originalSender, originalTarget, value, messageID):
        """
        Route an incoming answer to a message originally sent by this queue.
        """
        def txn():
            qm = self._messageFromSender(originalSender, messageID)
            if qm is None:
                return
            c = qm.consequence
            if c is not None:
                c.answerReceived(value, qm.value,
                                 qm.sender, qm.target)
            elif value.type == DELIVERY_ERROR:
                try:
                    raise MessageTransportError(value.data)
                except MessageTransportError:
                    log.err(Failure(),
                            "An unhandled delivery error occurred on a message"
                            " with no consequence.")
            qm.deleteFromStore()
        try:
            self.store.transact(txn)
        except:
            log.err(Failure(),
                    "An unhandled error occurred while handling a response to "
                    "an inter-store message.")
            def answerProcessingFailure():
                qm = self._messageFromSender(originalSender, messageID)
                _FailedAnswer.create(store=qm.store,
                                     consequence=qm.consequence,
                                     sender=originalSender,
                                     target=originalTarget,
                                     messageValue=qm.value,
                                     answerValue=value)
                qm.deleteFromStore()
            self.store.transact(answerProcessingFailure)
        return defer.succeed(None)