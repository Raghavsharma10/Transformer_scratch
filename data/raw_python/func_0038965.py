def summarize_classes(self):
        """
        Summary of classes: names, numeric labels and sizes

        Returns
        -------
        tuple : class_set, label_set, class_sizes

        class_set : list
            List of names of all the classes
        label_set : list
            Label for each class in class_set
        class_sizes : list
            Size of each class (number of samples)

        """

        class_sizes = np.zeros(len(self.class_set))
        for idx, cls in enumerate(self.class_set):
            class_sizes[idx] = self.class_sizes[cls]

        # TODO consider returning numeric label set e.g. for use in scikit-learn
        return self.class_set, self.label_set, class_sizes