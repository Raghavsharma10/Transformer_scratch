def transfers_complete(self):
        """Check if all transfers are completed."""

        for transfer in self.transfers:
            if not transfer.is_complete:
                error = {
                    'errorcode': 4003,
                    'errormessage': 'You must complete transfer before logout.'
                    }
                hellraiser(error)