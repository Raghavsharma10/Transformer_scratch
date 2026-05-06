def can_editions(self):
        """
        bool: :const:`True` if :attr:`address` can register the number of
        editions of :attr:`piece_address` else :const:`False`.

        In order to register the number of editions:

        1. There needs to a least one transaction for the :attr:`piece_address`
        (the registration of the master edition).

        2. A piece with address :attr:`piece_address` needs to be registered
        with ``'ASCRIBESPOOL01PIECE'`` (master edition).

        3. The number of editions should have not been set yet (no tx with
        verb ``'ASCRIBESPOOLEDITIONS'``).

        """
        chain = BlockchainSpider.chain(self._tree, REGISTERED_PIECE_CODE)

        if len(chain) == 0:
            self.reason = 'Master edition not yet registered'
            return False

        number_editions = chain[0]['number_editions']
        if number_editions != 0:
            self.reason = 'Number of editions was already registered for this piece'
            return False

        return True