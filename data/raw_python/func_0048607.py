def simple_spool_transaction(self, from_address, to, op_return, min_confirmations=6):
        """
        Utililty function to create the spool transactions. Selects the inputs,
        encodes the op_return and constructs the transaction.

        Args:
            from_address (str): Address originating the transaction
            to (str): list of addresses to receive tokens (file_hash, file_hash_metadata, ...)
            op_return (str): String representation of the spoolverb, as returned by the properties of Spoolverb
            min_confirmations (int): Number of confirmations when chosing the inputs of the transaction. Defaults to 6

        Returns:
            str: unsigned transaction

        """
        # list of addresses to send
        ntokens = len(to)
        nfees = old_div(self._t.estimate_fee(ntokens, 2), self.fee)
        inputs = self.select_inputs(from_address, nfees, ntokens, min_confirmations=min_confirmations)
        # outputs
        outputs = [{'address': to_address, 'value': self.token} for to_address in to]
        outputs += [{'script': self._t._op_return_hex(op_return), 'value': 0}]
        # build transaction
        unsigned_tx = self._t.build_transaction(inputs, outputs)
        return unsigned_tx