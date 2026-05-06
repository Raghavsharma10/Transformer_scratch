def in_feed(self, feed):
        '''Return the objects in the target feed'''
        kwargs = {self.model._rel_to_feed: feed}
        return self.filter(**kwargs)