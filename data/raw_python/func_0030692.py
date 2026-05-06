def build(self):
        """Access build configuration values as attributes. See self.process
            for a usage example"""
        from ambry.orm.config import BuildConfigGroupAccessor

        # It is a lightweight object, so no need to cache
        return BuildConfigGroupAccessor(self.dataset, 'buildstate', self._session)