def index_library_datasets(self, tick_f=None):
        """ Indexes all datasets of the library.

        Args:
            tick_f (callable, optional): callable of one argument. Gets string with index state.

        """

        dataset_n = 0
        partition_n = 0

        def tick(d, p):
            if tick_f:
                tick_f('datasets: {} partitions: {}'.format(d, p))

        for dataset in self.library.datasets:

            if self.backend.dataset_index.index_one(dataset):
                # dataset added to index
                dataset_n += 1
                tick(dataset_n, partition_n)
                for partition in dataset.partitions:
                    self.backend.partition_index.index_one(partition)
                    partition_n += 1
                    tick(dataset_n, partition_n)
            else:
                # dataset already indexed
                pass