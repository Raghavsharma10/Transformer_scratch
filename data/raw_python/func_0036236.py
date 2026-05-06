def is_complete(self):
        """:rtype: ``bool`` ``True`` if transfer is complete"""

        if 'status' in self.transfer_info:
            self._complete = self.transfer_info['status'] == 'STATUS_COMPLETE'

        return self._complete