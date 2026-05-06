def compute_training_sizes(train_perc, class_sizes, stratified=True):
    """Computes the maximum training size that the smallest class can provide """

    size_per_class = np.int64(np.around(train_perc * class_sizes))

    if stratified:
        print("Different classes in training set are stratified to match smallest class!")

        # per-class
        size_per_class = np.minimum(np.min(size_per_class), size_per_class)

        # single number
        reduced_sizes = np.unique(size_per_class)
        if len(reduced_sizes) != 1:  # they must all be the same
            raise ValueError("Error in stratification of training set based on "
                             "smallest class!")

    total_test_samples = np.int64(np.sum(class_sizes) - sum(size_per_class))

    return size_per_class, total_test_samples