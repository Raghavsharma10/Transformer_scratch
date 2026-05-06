def encrypt(self, paddedPlaintext):
        """
        :type paddedPlaintext: str
        """
        # TODO: make this less ugly and python 2 and 3 compatible
        # paddedMessage = bytearray(paddedMessage.encode() if (sys.version_info >= (3, 0) and not type(paddedMessage) in (bytes, bytearray)) or type(paddedMessage) is unicode else paddedMessage)
        if (sys.version_info >= (3, 0) and
                not type(paddedPlaintext) in (bytes, bytearray)) or type(paddedPlaintext) is unicode:
            paddedPlaintext = bytearray(paddedPlaintext.encode())
        else:
            paddedPlaintext = bytearray(paddedPlaintext)
        try:
            record = self.senderKeyStore.loadSenderKey(self.senderKeyName)
            senderKeyState = record.getSenderKeyState()
            senderKey = senderKeyState.getSenderChainKey().getSenderMessageKey()
            ciphertext = self.getCipherText(senderKey.getIv(), senderKey.getCipherKey(), paddedPlaintext)

            senderKeyMessage = SenderKeyMessage(senderKeyState.getKeyId(),
                                                senderKey.getIteration(),
                                                ciphertext,
                                                senderKeyState.getSigningKeyPrivate())

            senderKeyState.setSenderChainKey(senderKeyState.getSenderChainKey().getNext())
            self.senderKeyStore.storeSenderKey(self.senderKeyName, record)

            return senderKeyMessage.serialize()
        except InvalidKeyIdException as e:
            raise NoSessionException(e)