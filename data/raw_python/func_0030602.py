def _subset(self, subset):
        """Return a new pipeline with a subset of the sections"""
        pl = Pipeline(bundle=self.bundle)
        for group_name, pl_segment in iteritems(self):
            if group_name not in subset:
                continue
            pl[group_name] = pl_segment
        return pl