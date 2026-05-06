def can_unconsign(self):
        """
        bool: :const:`True` if :attr:`address` can unconsign the edition
        :attr:`edition_number` of :attr:`piece_address` else :const:`False`.

        If the last transaction is a consignment of the edition to the user.

        """
        chain = BlockchainSpider.chain(self._tree, self.edition_number)
        if len(chain) == 0:
            self.reason = 'Master edition not yet registered'
            return False

        chain = BlockchainSpider.strip_loan(chain)
        action = chain[-1]['action']
        piece_address = chain[-1]['piece_address']
        edition_number = chain[-1]['edition_number']
        to_address = chain[-1]['to_address']

        if action != 'CONSIGN' or piece_address != self.piece_address or edition_number != self.edition_number or to_address != self.address:
            self.reason = 'Edition number {} is not consigned to {}'.format(self.edition_number, self.address)
            return False

        return True