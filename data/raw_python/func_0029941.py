def tables(self):
        """ Return a iterator of tables in this bundle
        :return:
        """
        from ambry.orm import Table
        from sqlalchemy.orm import lazyload

        return (self.dataset.session.query(Table)
                .filter(Table.d_vid == self.dataset.vid)
                .options(lazyload('*'))
                .all())