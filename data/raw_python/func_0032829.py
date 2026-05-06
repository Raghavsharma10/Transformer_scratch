def messageRemote(self, cmdObj, consequence=None, **args):
        """
        Send a message to the peer identified by the target, via the
        given L{Command} object and arguments.

        @param cmdObj: a L{twisted.protocols.amp.Command}, whose serialized
        form will be the message.

        @param consequence: an L{IDeliveryConsequence} provider which will
        handle the result of this message (or None, if no response processing
        is desired).

        @param args: keyword arguments which match the C{cmdObj}'s arguments
        list.

        @return: L{None}
        """
        messageBox = cmdObj.makeArguments(args, self)
        messageBox[COMMAND] = cmdObj.commandName
        messageData = messageBox.serialize()
        self.queue.queueMessage(self.sender, self.target,
                                Value(AMP_MESSAGE_TYPE, messageData),
                                consequence)