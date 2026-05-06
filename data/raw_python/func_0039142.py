def append(self, dataset, identifier):
        """
        Adds a dataset, if compatible with the existing ones.

        Parameters
        ----------

        dataset : MLDataset or compatible

        identifier : hashable
            String or integer or another hashable to uniquely identify this dataset

        """

        dataset = dataset if isinstance(dataset, MLDataset) else MLDataset(dataset)

        if not self._is_init:
            self._ids = set(dataset.keys)
            self._classes = dataset.classes
            self._class_sizes = dataset.class_sizes

            self._num_samples = len(self._ids)
            self._modalities[identifier] = dataset.data
            self._num_features.append(dataset.num_features)

            # maintaining a no-data MLDataset internally for reuse its methods
            self._dataset = copy(dataset)
            # replacing its data with zeros
            self._dataset.data = {id_: np.zeros(1) for id_ in self._ids}

            self._is_init = True
        else:
            # this also checks for the size (num_samples)
            if set(dataset.keys) != self._ids:
                raise ValueError('Differing set of IDs in two datasets.'
                                 'Unable to add this dataset to the MultiDataset.')

            if dataset.classes != self._classes:
                raise ValueError('Classes for IDs differ in the two datasets.')

            if identifier not in self._modalities:
                self._modalities[identifier] = dataset.data
                self._num_features.append(dataset.num_features)
            else:
                raise KeyError('{} already exists in MultiDataset'.format(identifier))

        # each addition should be counted, if successful
        self._modality_count += 1