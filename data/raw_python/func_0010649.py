def merge(self, dataset):
        """ Merge the specified dataset on top of the existing data.

        This replaces all values in the existing dataset with the values from the
        given dataset.

        Args:
            dataset (TaskData): A reference to the TaskData object that should be merged
                on top of the existing object.
        """
        def merge_data(source, dest):
            for key, value in source.items():
                if isinstance(value, dict):
                    merge_data(value, dest.setdefault(key, {}))
                else:
                    dest[key] = value
            return dest

        merge_data(dataset.data, self._data)

        for h in dataset.task_history:
            if h not in self._task_history:
                self._task_history.append(h)