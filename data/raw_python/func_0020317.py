def bind(self, callback, sender=None):
        '''Bind a ``callback`` for a given ``sender``.'''
        key = (_make_id(callback), _make_id(sender))
        self.callbacks.append((key, callback))