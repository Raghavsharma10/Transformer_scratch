def _update_names(self):
        """Update the derived names"""

        d = dict(
            table=self.table_name,
            time=self.time,
            space=self.space,
            grain=self.grain,
            variant=self.variant,
            segment=self.segment
        )

        assert self.dataset

        name = PartialPartitionName(**d).promote(self.dataset.identity.name)

        self.name = str(name.name)
        self.vname = str(name.vname)
        self.cache_key = name.cache_key
        self.fqname = str(self.identity.fqname)