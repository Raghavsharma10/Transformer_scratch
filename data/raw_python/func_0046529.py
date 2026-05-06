def can_register(self):
        """
        bool: :const:`True` if :attr:`address` can register the edition
        :attr:`edition_number` of :attr:`piece_address` else :const:`False`.

        In order to register an edition:

        1. The master piece needs to be registered.
        2. The number of editions needs to be registered.
        3. The :attr:`edition_number` should not have been registered yet.

        .. todo:: Also check that the root address owns the piece.
            Right now we cannot do this because we only receive
            the leaf address when registering an edition.

        """
        chain = BlockchainSpider.chain(self._tree, REGISTERED_PIECE_CODE)

        # edition 0 should only have two transactions: REGISTER and EDITIONS
        if len(chain) == 0:
            self.reason = 'Master edition not yet registered'
            return False

        chain = BlockchainSpider.strip_loan(chain)
        number_editions = chain[0]['number_editions']
        if number_editions == 0:
            self.reason = 'Number of editions not yet registered'
            return False

        if self.edition_number > number_editions:
            self.reason = 'You can only register {} editions. You are trying to register edition {}'.format(number_editions, self.edition_number)
            return False

        if self.edition_number in self._tree:
            self.reason = 'Edition number {} is already registered in the blockchain'. format(self.edition_number)
            return False

        return True