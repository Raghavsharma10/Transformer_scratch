def _boxFromData(self, messageData):
        """
        A box.

        @param messageData: a serialized AMP box representing either a message
        or an error.
        @type messageData: L{str}

        @raise MalformedMessage: if the C{messageData} parameter does not parse
        to exactly one AMP box.
        """
        inputBoxes = parseString(messageData)
        if not len(inputBoxes) == 1:
            raise MalformedMessage()
        [inputBox] = inputBoxes
        return inputBox