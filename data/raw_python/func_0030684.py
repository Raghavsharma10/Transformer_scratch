def update_done(self, *args, **kwargs):
        """Clear out the previous update"""
        kwargs['state'] = 'done'
        self.update(*args, **kwargs)
        self.rec = None