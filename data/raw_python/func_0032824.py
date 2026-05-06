def _messageFromSender(self, sender, messageID):
        """
        Locate a previously queued message by a given sender and messageID.
        """
        return self.store.findUnique(
            _QueuedMessage,
            AND(_QueuedMessage.senderUsername == sender.localpart,
                _QueuedMessage.senderDomain == sender.domain,
                _QueuedMessage.messageID == messageID),
            default=None)