def partition(self, ref, localize=False):
        """ Finds partition by ref and converts to bundle partition.

        :param ref: A partition reference
        :param localize: If True, copy a remote partition to local filesystem. Defaults to False
        :raises: NotFoundError: if partition with given ref not found.
        :return: orm.Partition: found partition.
        """

        if not ref:
            raise NotFoundError("No partition for empty ref")

        try:
            on = ObjectNumber.parse(ref)
            ds_on = on.as_dataset

            ds = self._db.dataset(ds_on)  # Could do it in on SQL query, but this is easier.

            # The refresh is required because in some places the dataset is loaded without the partitions,
            # and if that persist, we won't have partitions in it until it is refreshed.

            self.database.session.refresh(ds)

            p = ds.partition(ref)

        except NotObjectNumberError:
            q = (self.database.session.query(Partition)
                 .filter(or_(Partition.name == str(ref), Partition.vname == str(ref)))
                 .order_by(Partition.vid.desc()))

            p = q.first()

        if not p:
            raise NotFoundError("No partition for ref: '{}'".format(ref))

        b = self.bundle(p.d_vid)
        p = b.wrap_partition(p)

        if localize:
            p.localize()

        return p