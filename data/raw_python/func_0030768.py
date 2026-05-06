def _find_orm(self, pnq):
        """Return a Partition object from the database based on a PartitionId.

        An ORM object is returned, so changes can be persisted.

        """
        # import sqlalchemy.orm.exc
        from ambry.orm import Partition as OrmPartition  # , Table
        from sqlalchemy.orm import joinedload  # , joinedload_all

        assert isinstance(pnq, PartitionNameQuery), "Expected PartitionNameQuery, got {}".format(type(pnq))

        pnq = pnq.with_none()

        q = self.bundle.dataset._database.session.query(OrmPartition)

        if pnq.fqname is not NameQuery.ANY:
            q = q.filter(OrmPartition.fqname == pnq.fqname)
        elif pnq.vname is not NameQuery.ANY:
            q = q.filter(OrmPartition.vname == pnq.vname)
        elif pnq.name is not NameQuery.ANY:
            q = q.filter(OrmPartition.name == str(pnq.name))
        else:
            if pnq.time is not NameQuery.ANY:
                q = q.filter(OrmPartition.time == pnq.time)

            if pnq.space is not NameQuery.ANY:
                q = q.filter(OrmPartition.space == pnq.space)

            if pnq.grain is not NameQuery.ANY:
                q = q.filter(OrmPartition.grain == pnq.grain)

            if pnq.format is not NameQuery.ANY:
                q = q.filter(OrmPartition.format == pnq.format)

            if pnq.segment is not NameQuery.ANY:
                q = q.filter(OrmPartition.segment == pnq.segment)

            if pnq.table is not NameQuery.ANY:

                if pnq.table is None:
                    q = q.filter(OrmPartition.t_id is None)
                else:
                    tr = self.bundle.table(pnq.table)

                    if not tr:
                        raise ValueError("Didn't find table named {} in {} bundle path = {}".format(
                            pnq.table, pnq.vname, self.bundle.database.path))

                    q = q.filter(OrmPartition.t_vid == tr.vid)

        ds = self.bundle.dataset

        q = q.filter(OrmPartition.d_vid == ds.vid)

        q = q.order_by(
            OrmPartition.vid.asc()).order_by(
            OrmPartition.segment.asc())

        q = q.options(joinedload(OrmPartition.table))

        return q