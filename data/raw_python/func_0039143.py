def holdout(self,
                train_perc=0.7,
                num_rep=50,
                stratified=True,
                return_ids_only=False,
                format='MLDataset'):
        """
        Builds a generator for train and test sets for cross-validation.

        """

        ids_in_class = {cid: self._dataset.sample_ids_in_class(cid)
                        for cid in self._class_sizes.keys()}

        sizes_numeric = np.array([len(ids_in_class[cid]) for cid in ids_in_class.keys()])
        size_per_class, total_test_count = compute_training_sizes(
                train_perc, sizes_numeric, stratified=stratified)

        if len(self._class_sizes) != len(size_per_class):
            raise ValueError('size spec differs in num elements with class sizes!')

        for rep in range(num_rep):
            print('rep {}'.format(rep))

            train_set = list()
            for index, (cls_id, class_size) in enumerate(self._class_sizes.items()):
                # shuffling the IDs each time
                random.shuffle(ids_in_class[cls_id])

                subset_size = max(0, min(class_size, size_per_class[index]))
                if subset_size < 1 or class_size < 1:
                    warnings.warn('No subjects from class {} were selected.'
                                  ''.format(cls_id))
                else:
                    subsets_this_class = ids_in_class[cls_id][0:size_per_class[index]]
                    train_set.extend(subsets_this_class)

            # this ensures both are mutually exclusive!
            test_set = list(self._ids - set(train_set))

            if return_ids_only:
                # when only IDs are required, without associated features
                # returning tuples to prevent accidental changes
                yield tuple(train_set), tuple(test_set)
            else:
                yield self._get_data(train_set, format), self._get_data(test_set, format)