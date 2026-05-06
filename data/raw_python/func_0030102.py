def dataset(self, ref, load_all=False, exception=True):
        """Return all datasets"""
        return self.database.dataset(ref, load_all=load_all, exception=exception)