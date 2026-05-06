def create(self, senderKeyName):
        """
        :type senderKeyName: SenderKeyName
        """
        try:
            senderKeyRecord = self.senderKeyStore.loadSenderKey(senderKeyName);

            if senderKeyRecord.isEmpty() :
                senderKeyRecord.setSenderKeyState(KeyHelper.generateSenderKeyId(),
                                                0,
                                                KeyHelper.generateSenderKey(),
                                                KeyHelper.generateSenderSigningKey());
                self.senderKeyStore.storeSenderKey(senderKeyName, senderKeyRecord);

            state = senderKeyRecord.getSenderKeyState();

            return SenderKeyDistributionMessage(state.getKeyId(),
                                                state.getSenderChainKey().getIteration(),
                                                state.getSenderChainKey().getSeed(),
                                                state.getSigningKeyPublic());
        except (InvalidKeyException, InvalidKeyIdException) as e:
            raise AssertionError(e)