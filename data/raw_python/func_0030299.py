def update(self, rec=None, drop=None, tables=None, install=None, materialize=None,
               indexes=None, joins=0, views=0):
        """ Updates current record.

        Args:
            rec (FIMRecord):
        """
        if not drop:
            drop = []

        if not tables:
            tables = set()

        if not install:
            install = set()

        if not materialize:
            materialize = set()

        if not indexes:
            indexes = set()

        if rec:
            self.update(
                drop=rec.drop, tables=rec.tables, install=rec.install, materialize=rec.materialize,
                indexes=rec.indexes, joins=rec.joins
            )

        self.drop += drop
        self.tables |= set(tables)
        self.install |= set(install)
        self.materialize |= set(materialize)
        self.indexes |= set(indexes)

        self.joins += joins
        self.views += views

        # Joins or views promote installed partitions to materialized partitions
        if self.joins > 0 or self.views > 0:
            self.materialize |= self.install
            self.install = set()