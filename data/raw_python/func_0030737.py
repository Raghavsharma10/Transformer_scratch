def new_dataset(self, *args, **kwargs):
        """ Creates a new dataset

        :param args: Positional args passed to the Dataset constructor.
        :param kwargs:  Keyword args passed to the Dataset constructor.
        :return: :class:`ambry.orm.Dataset`
        :raises: :class:`ambry.orm.ConflictError` if the a Dataset records already exists with the given vid
        """

        ds = Dataset(*args, **kwargs)

        try:
            self.session.add(ds)
            self.session.commit()
            ds._database = self
            return ds
        except IntegrityError as e:
            self.session.rollback()
            raise ConflictError(
                "Can't create dataset '{}'; one probably already exists: {} ".format(str(ds), e))