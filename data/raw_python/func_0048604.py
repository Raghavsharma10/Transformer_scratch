def register_piece(self, from_address, to_address, hash, password, min_confirmations=6, sync=False, ownership=True):
        """
        Register a piece

        Args:
            from_address (Tuple[str]): Federation address. All register transactions
                originate from the the Federation wallet
            to_address (str): Address registering the edition
            hash (Tuple[str]): Hash of the piece. (file_hash, file_hash_metadata)
            password (str): Federation wallet password. For signing the transaction
            edition_num (int): The number of the edition to register. User
                edition_num=0 to register the master edition
            min_confirmations (int): Override the number of confirmations when
                chosing the inputs of the transaction. Defaults to 6
            sync (bool): Perform the transaction in synchronous mode, the call to the
                function will block until there is at least on confirmation on
                the blockchain. Defaults to False
            ownership (bool): Check ownsership in the blockchain before pushing the
                transaction. Defaults to True

        Returns:
            str: transaction id

        """
        file_hash, file_hash_metadata = hash
        path, from_address = from_address
        verb = Spoolverb()
        unsigned_tx = self.simple_spool_transaction(from_address,
                                                    [file_hash, file_hash_metadata, to_address],
                                                    op_return=verb.piece,
                                                    min_confirmations=min_confirmations)

        signed_tx = self._t.sign_transaction(unsigned_tx, password)
        txid = self._t.push(signed_tx)
        return txid