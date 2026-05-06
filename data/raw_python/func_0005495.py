def _prepare_data(self):
        """Formats the data in the current transaction for processing"""
        total_amount = self.total_amount or self.calculate_total_amt()
        self._data = {
            "invoice": {
                "items": self.__encode_items(self.items),
                "taxes": self.taxes,
                "total_amount": total_amount,
                "description": self.description,
                "channels": self.channels
            },
            "store": self.store.info,
            "custom_data": self.custom_data,
            "actions": {
                "cancel_url": self.cancel_url,
                "return_url": self.return_url,
                "callback_url": self.callback_url
            }
        }
        return self._data