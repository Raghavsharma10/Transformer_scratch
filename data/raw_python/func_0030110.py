def checkin_bundle(self, db_path, replace=True, cb=None):
        """Add a bundle, as a Sqlite file, to this library"""
        from ambry.orm.exc import NotFoundError

        db = Database('sqlite:///{}'.format(db_path))
        db.open()

        if len(db.datasets) == 0:
            raise NotFoundError("Did not get a dataset in the {} bundle".format(db_path))


        ds = db.dataset(db.datasets[0].vid)  # There should only be one

        assert ds is not None
        assert ds._database

        try:
            b = self.bundle(ds.vid)
            self.logger.info(
                "Removing old bundle before checking in new one of same number: '{}'"
                .format(ds.vid))
            self.remove(b)
        except NotFoundError:
            pass

        try:
            self.dataset(ds.vid)  # Skip loading bundles we already have
        except NotFoundError:
            self.database.copy_dataset(ds, cb=cb)

        b = self.bundle(ds.vid)  # It had better exist now.
        # b.state = Bundle.STATES.INSTALLED
        b.commit()

        #self.search.index_library_datasets(tick)

        self.search.index_bundle(b)

        return b