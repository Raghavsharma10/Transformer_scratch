def save(self, *args, **kwargs):
        """Sync payment profile on Authorize.NET if sync kwarg is not False"""
        if kwargs.pop('sync', True):
            self.push_to_server()
        self.card_code = None
        self.card_number = "XXXX%s" % self.card_number[-4:]
        super(CustomerPaymentProfile, self).save(*args, **kwargs)