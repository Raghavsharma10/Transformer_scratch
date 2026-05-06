def shelve(self):
        """Return an opened shelve object.

        """
        logger.info('creating shelve data')
        fname = str(self.create_path.absolute())
        inst = sh.open(fname, writeback=self.writeback)
        self.is_open = True
        return inst