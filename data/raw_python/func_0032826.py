def queueMessage(self, sender, target, value,
                     consequence=None):
        """
        Queue a persistent outgoing message.

        @param sender: The a description of the shared item that is the sender
        of the message.
        @type sender: L{xmantissa.sharing.Identifier}

        @param target: The a description of the shared item that is the target
        of the message.
        @type target: L{xmantissa.sharing.Identifier}

        @param consequence: an item stored in the same database as this
        L{MessageQueue} implementing L{IDeliveryConsequence}.
        """
        self.messageCounter += 1
        _QueuedMessage.create(store=self.store,
                              sender=sender,
                              target=target,
                              value=value,
                              messageID=self.messageCounter,
                              consequence=consequence)
        self._scheduleMePlease()