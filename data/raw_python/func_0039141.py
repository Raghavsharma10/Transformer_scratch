def _load(self, dataset_spec):
        """Actual loading of datasets"""

        for idx, ds in enumerate(dataset_spec):
            self.append(ds, idx)