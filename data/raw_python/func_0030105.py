def bundle(self, ref, capture_exceptions=False):
        """Return a bundle build on a dataset, with the given vid or id reference"""
        from ..orm.exc import NotFoundError

        if isinstance(ref, Dataset):
            ds = ref
        else:
            try:
                ds = self._db.dataset(ref)
            except NotFoundError:
                ds = None

        if not ds:
            try:
                p = self.partition(ref)
                ds = p._bundle.dataset
            except NotFoundError:
                ds = None

        if not ds:
            raise NotFoundError('Failed to find dataset for ref: {}'.format(ref))

        b = Bundle(ds, self)
        b.capture_exceptions = capture_exceptions

        return b