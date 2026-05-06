def partition(self, ref=None, **kwargs):
        """ Returns partition by ref. """
        from .exc import NotFoundError
        from six import text_type

        if ref:

            for p in self.partitions: # This is slow for large datasets, like Census years.
                if (text_type(ref) == text_type(p.name) or text_type(ref) == text_type(p.id) or
                            text_type(ref) == text_type(p.vid)):
                    return p

            raise NotFoundError("Failed to find partition for ref '{}' in dataset '{}'".format(ref, self.name))

        elif kwargs:
            from ..identity import PartitionNameQuery

            pnq = PartitionNameQuery(**kwargs)
            return self._find_orm