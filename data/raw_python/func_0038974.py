def random_subset_ids(self, perc_per_class=0.5):
        """
        Returns a random subset of sample ids (size in percentage) within each class.

        Parameters
        ----------
        perc_per_class : float
            Fraction of samples per class

        Returns
        -------
        subset : list
            Combined list of sample ids from all classes.

        Raises
        ------
        ValueError
            If no subjects from one or more classes were selected.
        UserWarning
            If an empty or full dataset is requested.

        """

        class_sizes = self.class_sizes
        subsets = list()

        if perc_per_class <= 0.0:
            warnings.warn('Zero percentage requested - returning an empty dataset!')
            return list()
        elif perc_per_class >= 1.0:
            warnings.warn('Full or a larger dataset requested - returning a copy!')
            return self.keys

        # seeding the random number generator
        # random.seed(random_seed)

        for class_id, class_size in class_sizes.items():
            # samples belonging to the class
            this_class = self.keys_with_value(self.classes, class_id)
            # shuffling the sample order; shuffling works in-place!
            random.shuffle(this_class)
            # calculating the requested number of samples
            subset_size_this_class = np.int64(np.floor(class_size * perc_per_class))
            # clipping the range to [1, n]
            subset_size_this_class = max(1, min(class_size, subset_size_this_class))
            if subset_size_this_class < 1 or len(this_class) < 1 or this_class is None:
                # warning if none were selected
                raise ValueError(
                    'No subjects from class {} were selected.'.format(class_id))
            else:
                subsets_this_class = this_class[0:subset_size_this_class]
                subsets.extend(subsets_this_class)

        if len(subsets) > 0:
            return subsets
        else:
            warnings.warn('Zero samples were selected. Returning an empty list!')
            return list()