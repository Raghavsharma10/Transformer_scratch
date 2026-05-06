def refill_main_wallet(self, from_address, to_address, nfees, ntokens, password, min_confirmations=6, sync=False):
        """
        Refill the Federation wallet with tokens and fees. This keeps the federation wallet clean.
        Dealing with exact values simplifies the transactions. No need to calculate change. Easier to keep track of the
        unspents and prevent double spends that would result in transactions being rejected by the bitcoin network.

        Args:

            from_address (Tuple[str]): Refill wallet address. Refills the federation wallet with tokens and fees
            to_address (str): Federation wallet address
            nfees (int): Number of fees to transfer. Each fee is 10000 satoshi. Used to pay for the transactions
            ntokens (int): Number of tokens to transfer. Each token is 600 satoshi. Used to register hashes in the blockchain
            password (str): Password for the Refill wallet. Used to sign the transaction
            min_confirmations (int): Number of confirmations when chosing the inputs of the transaction. Defaults to 6
            sync (bool): Perform the transaction in synchronous mode, the call to the function will block until there is at
                least on confirmation on the blockchain. Defaults to False

        Returns:
            str: transaction id
        """
        path, from_address = from_address
        unsigned_tx = self._t.simple_transaction(from_address,
                                                 [(to_address, self.fee)] * nfees + [(to_address, self.token)] * ntokens,
                                                 min_confirmations=min_confirmations)

        signed_tx = self._t.sign_transaction(unsigned_tx, password)
        txid = self._t.push(signed_tx)
        return txid