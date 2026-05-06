def save(self, *args, **kwargs):
        """If creating new instance, create profile on Authorize.NET also"""
        data = kwargs.pop('data', {})
        sync = kwargs.pop('sync', True)
        if not self.id and sync:
            self.push_to_server(data)
        super(CustomerProfile, self).save(*args, **kwargs)