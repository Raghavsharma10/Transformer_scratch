def dataset(self, ref, load_all=False, exception=True):
        """Return a dataset, given a vid or id

        :param ref: Vid or id  for a dataset. If an id is provided, will it will return the one with the
        largest revision number
        :param load_all: Use a query that eagerly loads everything.
        :return: :class:`ambry.orm.Dataset`

        """

        ref = str(ref)

        try:
            ds = self.session.query(Dataset).filter(Dataset.vid == ref).one()
        except NoResultFound:
            ds = None

        if not ds:
            try:
                ds = self.session \
                    .query(Dataset) \
                    .filter(Dataset.id == ref) \
                    .order_by(Dataset.revision.desc()) \
                    .first()
            except NoResultFound:
                ds = None

        if not ds:
            try:
                ds = self.session.query(Dataset).filter(Dataset.vname == ref).one()
            except NoResultFound:
                ds = None

        if not ds:
            try:
                ds = self.session \
                    .query(Dataset) \
                    .filter(Dataset.name == ref) \
                    .order_by(Dataset.revision.desc()) \
                    .first()
            except NoResultFound:
                ds = None

        if ds:
            ds._database = self
            return ds
        elif exception:
            raise NotFoundError('No dataset in library for vid : {} '.format(ref))
        else:
            return None