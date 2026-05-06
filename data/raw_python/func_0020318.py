def fire(self, sender=None, **params):
        '''Fire callbacks from a ``sender``.'''
        keys = (_make_id(None), _make_id(sender))
        results = []
        for (_, key), callback in self.callbacks:
            if key in keys:
                results.append(callback(self, sender, **params))
        return results