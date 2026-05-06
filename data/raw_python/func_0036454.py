def _restore_transfers(self, response):
        """Restore transfers from josn retreived Filemail
        :param response: response object from request
        :rtype: ``list`` with :class:`Transfer` objects
        """

        transfers = []
        for transfer_data in response.json()['transfers']:
            transfer = Transfer(self, _restore=True)
            transfer.transfer_info.update(transfer_data)
            transfer.get_files()
            transfers.append(transfer)

        return transfers