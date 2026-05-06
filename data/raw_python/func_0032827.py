def run(self):
        """
        Attmept to deliver the first outgoing L{QueuedMessage}; return a time
        to reschedule if there are still more retries or outgoing messages to
        send.
        """
        delay = None
        router = self.siteRouter
        for qmsg in self.store.query(_QueuedMessage,
                                     sort=_QueuedMessage.storeID.ascending):
            try:
                self._verifySender(qmsg.sender)
            except:
                self.routeAnswer(qmsg.sender, qmsg.target,
                                 Value(DELIVERY_ERROR, ERROR_BAD_SENDER),
                                 qmsg.messageID)
                log.err(Failure(),
                        "Could not verify sender for sending message.")
            else:
                router.routeMessage(qmsg.sender, qmsg.target,
                                    qmsg.value, qmsg.messageID)

        for answer in self.store.query(_AlreadyAnswered,
                                       sort=_AlreadyAnswered.storeID.ascending):
            self._deliverAnswer(answer)
        nextmsg = self.store.findFirst(_QueuedMessage, default=None)
        if nextmsg is not None:
            delay = _RETRANSMIT_DELAY
        else:
            nextanswer = self.store.findFirst(_AlreadyAnswered, default=None)
            if nextanswer is not None:
                delay = _RETRANSMIT_DELAY
        if delay is not None:
            return IScheduler(self.store).now() + timedelta(seconds=delay)