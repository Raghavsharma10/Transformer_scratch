def partition(self, id_):
        """Get a partition by the id number.

        Arguments:
            id_ -- a partition id value

        Returns:
            A partitions.Partition object

        Throws:
            a Sqlalchemy exception if the partition either does not exist or
            is not unique

        Because this method works on the bundle, the id_ ( without version information )
        is equivalent to the vid ( with version information )

        """
        from ..orm import Partition as OrmPartition
        from sqlalchemy import or_
        from ..identity import PartialPartitionName

        if isinstance(id_, PartitionIdentity):
            id_ = id_.id_
        elif isinstance(id_, PartialPartitionName):
            id_ = id_.promote(self.bundle.identity.name)

        session = self.bundle.dataset._database.session
        q = session\
            .query(OrmPartition)\
            .filter(OrmPartition.d_vid == self.bundle.dataset.vid)\
            .filter(or_(OrmPartition.id == str(id_).encode('ascii'),
                        OrmPartition.vid == str(id_).encode('ascii')))

        try:
            orm_partition = q.one()
            return self.bundle.wrap_partition(orm_partition)
        except NoResultFound:
            orm_partition = None

        if not orm_partition:
            q = session\
                .query(OrmPartition)\
                .filter(OrmPartition.d_vid == self.bundle.dataset.vid)\
                .filter(OrmPartition.name == str(id_).encode('ascii'))

            try:
                orm_partition = q.one()
                return self.bundle.wrap_partition(orm_partition)
            except NoResultFound:
                orm_partition = None

        return orm_partition