def sync(self, data):
        """Overwrite local customer payment profile data with remote data"""
        for k, v in data.get('billing', {}).items():
            setattr(self, k, v)
        self.card_number = data.get('credit_card', {}).get('card_number',
                                                           self.card_number)
        self.save(sync=False)