def answerReceived(self, value, originalValue,
                       originalSender, originalTarget):
        """
        An answer was received.  Dispatch to the appropriate answer responder,
        i.e. a method on this object exposed with L{answerMethod.expose}.

        @see IDeliveryConsequence.answerReceived
        """
        if value.type != AMP_ANSWER_TYPE:
            raise UnknownMessageType()
        commandName = self._boxFromData(originalValue.data)[COMMAND]
        rawArgs = self._boxFromData(value.data)
        placeholder = _ProtocolPlaceholder(originalSender, originalTarget)
        if ERROR in rawArgs:
            thunk = errorMethod.errbackForName(self, commandName, rawArgs[ERROR])
            thunk(Failure(thunk.exception()))
        else:
            thunk = answerMethod.responderForName(self, commandName)
            arguments = thunk.command.parseResponse(rawArgs, placeholder)
            thunk(**arguments)