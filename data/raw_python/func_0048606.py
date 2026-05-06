def refill(self, from_address, to_address, nfees, ntokens, password, min_confirmations=6, sync=False):
        """
        Refill wallets with the necessary fuel to perform spool transactions

        Args:
            from_address (Tuple[str]): Federation wallet address. Fuels the wallets with tokens and fees. All transactions to wallets
                holding a particular piece should come from the Federation wallet
            to_address (str): Wallet address that needs to perform a spool transaction
            nfees (int): Number of fees to transfer. Each fee is 10000 satoshi. Used to pay for the transactions
            ntokens (int): Number of tokens to transfer. Each token is 600 satoshi. Used to register hashes in the blockchain
            password (str): Password for the Federation wallet. Used to sign the transaction
            min_confirmations (int): Number of confirmations when chosing the inputs of the transaction. Defaults to 6
            sync (bool): Perform the transaction in synchronous mode, the call to the function will block until there is at
                least on confirmation on the blockchain. Defaults to False

        Returns:
            str: transaction id

        """
        path, from_address = from_address
        verb = Spoolverb()
        # nfees + 1: nfees to refill plus one fee for the refill transaction itself
        inputs = self.select_inputs(from_address, nfees + 1, ntokens, min_confirmations=min_confirmations)
        outputs = [{'address': to_address, 'value': self.token}] * ntokens
        outputs += [{'address': to_address, 'value': self.fee}] * nfees
        outputs += [{'script': self._t._op_return_hex(verb.fuel), 'value': 0}]
        unsigned_tx = self._t.build_transaction(inputs, outputs)
        signed_tx = self._t.sign_transaction(unsigned_tx, password, path=path)
        txid = self._t.push(signed_tx)
        return txid