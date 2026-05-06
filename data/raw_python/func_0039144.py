def _get_data(self, id_list, format='MLDataset'):
        """Returns the data, from all modalities, for a given list of IDs"""

        format = format.lower()

        features = list()  # returning a dict would be better if AutoMKL() can handle it
        for modality, data in self._modalities.items():
            if format in ('ndarray', 'data_matrix'):
                # turning dict of arrays into a data matrix
                # this is arguably worse, as labels are difficult to pass
                subset = np.array(itemgetter(*id_list)(data))
            elif format in ('mldataset', 'pyradigm'):
                # getting container with fake data
                subset = self._dataset.get_subset(id_list)
                # injecting actual features
                subset.data = { id_: data[id_] for id_ in id_list }
            else:
                raise ValueError('Invalid output format - choose only one of '
                                 'MLDataset or data_matrix')

            features.append(subset)

        return features