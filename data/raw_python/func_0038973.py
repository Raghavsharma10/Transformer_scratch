def random_subset_ids_by_count(self, count_per_class=1):
        """
        Returns a random subset of sample ids of specified size by count,
            within each class.

        Parameters
        ----------
        count_per_class : int
            Exact number of samples per each class.

        Returns
        -------
        subset : list
            Combined list of sample ids from all classes.

        """

        class_sizes = self.class_sizes
        subsets = list()

        if count_per_class < 1:
            warnings.warn('Atleast one sample must be selected from each class')
            return list()
        elif count_per_class >= self.num_samples:
            warnings.warn('All samples requested - returning a copy!')
            return self.keys

        # seeding the random number generator
        # random.seed(random_seed)

        for class_id, class_size in class_sizes.items():
            # samples belonging to the class
            this_class = self.keys_with_value(self.classes, class_id)
            # shuffling the sample order; shuffling works in-place!
            random.shuffle(this_class)

            # clipping the range to [0, class_size]
            subset_size_this_class = max(0, min(class_size, count_per_class))
            if subset_size_this_class < 1 or this_class is None:
                # warning if none were selected
                warnings.warn('No subjects from class {} were selected.'.format(class_id))
            else:
                subsets_this_class = this_class[0:count_per_class]
                subsets.extend(subsets_this_class)

        if len(subsets) > 0:
            return subsets
        else:
            warnings.warn('Zero samples were selected. Returning an empty list!')
            return list()