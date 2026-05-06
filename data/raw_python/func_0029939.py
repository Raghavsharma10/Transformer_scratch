def partition(self, ref=None, **kwargs):
        """Return a partition in this bundle for a vid reference or name parts"""
        from ambry.orm.exc import NotFoundError
        from sqlalchemy.orm.exc import NoResultFound

        if not ref and not kwargs:
            return None

        if ref:
            for p in self.partitions:
                if ref == p.name or ref == p.vname or ref == p.vid or ref == p.id:
                  p._bundle = self
                  return p

            raise NotFoundError("No partition found for '{}' (a)".format(ref))

        elif kwargs:
            from ..identity import PartitionNameQuery
            pnq = PartitionNameQuery(**kwargs)
            try:
                p = self.partitions._find_orm(pnq).one()
                if p:
                    p._bundle = self
                    return p
            except NoResultFound:
                raise NotFoundError("No partition found for '{}' (b)".format(kwargs))