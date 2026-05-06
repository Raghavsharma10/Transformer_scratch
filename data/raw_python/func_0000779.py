def delete(self, **kwargs):
        """ Explicitly reload context with DB usage to get access
        to complete DB object.
        """
        self.reload_context(es_based=False, **kwargs)
        return super(ESCollectionView, self).delete(**kwargs)