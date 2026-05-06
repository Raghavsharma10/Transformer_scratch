def to_dict(self):
        """Returns attributes formatted as a dictionary."""
        d = {'id': self.id, 'classes': self.classes}
        d.update(self.kvs)
        return d