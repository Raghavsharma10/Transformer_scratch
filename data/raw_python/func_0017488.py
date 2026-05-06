def delete_instance(self, *args, **kwargs):
        """Send signals."""
        self.pre_delete.send(self)
        super(Model, self).delete_instance(*args, **kwargs)
        self.post_delete.send(self)