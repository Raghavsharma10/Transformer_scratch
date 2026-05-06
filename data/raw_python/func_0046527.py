def can_transfer(self):
        """
        bool: :const:`True` if :attr:`address` can transfer the edition
        :attr:`edition_number` of :attr:`piece_address` else :const:`False`.

        """
        # 1. The address needs to own the edition
        chain = BlockchainSpider.chain(self._tree, self.edition_number)

        if len(chain) == 0:
            self.reason = 'The edition number {} does not exist in the blockchain'.format(self.edition_number)
            return False

        chain = BlockchainSpider.strip_loan(chain)
        to_address = chain[-1]['to_address']
        if to_address != self.address:
            self.reason = 'Address {} does not own the edition number {}'.format(self.address, self.edition_number)
            return False

        return True