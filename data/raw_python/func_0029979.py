def clean_partitions(self):
        """Delete partition records and any built partition files. """
        import shutil
        from ambry.orm import ColumnStat

        # FIXME. There is a problem with the cascades for ColumnStats that prevents them from
        # being  deleted with the partitions. Probably, they are seen to be owed by the columns instead.
        self.session.query(ColumnStat).filter(ColumnStat.d_vid == self.dataset.vid).delete()

        self.dataset.delete_partitions()

        for s in self.sources:
            s.state = None

        if self.build_partition_fs.exists:
            try:
                shutil.rmtree(self.build_partition_fs.getsyspath('/'))
            except NoSysPathError:
                pass