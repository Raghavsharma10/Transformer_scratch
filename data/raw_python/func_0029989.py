def _reset_build(self, sources):
        """Remove partition datafiles and reset the datafiles to the INGESTED state"""
        from ambry.orm.exc import NotFoundError

        for p in self.dataset.partitions:
            if p.type == p.TYPE.SEGMENT:
                self.log("Removing old segment partition: {}".format(p.identity.name))
                try:
                    self.wrap_partition(p).local_datafile.remove()
                    self.session.delete(p)
                except NotFoundError:
                    pass

        for s in sources:

            # Don't delete partitions fro mother bundles!
            if s.reftype == 'partition':
                continue

            p = s.partition
            if p:
                try:
                    self.wrap_partition(p).local_datafile.remove()
                    self.session.delete(p)
                except NotFoundError:
                    pass

            if s.state in (self.STATES.BUILDING, self.STATES.BUILT):
                s.state = self.STATES.INGESTED

        self.commit()