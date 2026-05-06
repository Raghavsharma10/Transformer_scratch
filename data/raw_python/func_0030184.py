def index_bundle(self, bundle, force=False):
        """
        Indexes a bundle/dataset and all of its partitions
        :param bundle: A bundle or dataset object
        :param force: If true, index the document even if it already exists
        :return:
        """
        from ambry.orm.dataset import Dataset

        dataset = bundle if isinstance(bundle, Dataset) else bundle.dataset

        self.index_dataset(dataset, force)

        for partition in dataset.partitions:
            self.index_partition(partition, force)