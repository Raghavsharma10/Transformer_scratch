def __validate(data, classes, labels):
        "Validator of inputs."

        if not isinstance(data, dict):
            raise TypeError(
                'data must be a dict! keys: sample ID or any unique identifier')
        if not isinstance(labels, dict):
            raise TypeError(
                'labels must be a dict! keys: sample ID or any unique identifier')
        if classes is not None:
            if not isinstance(classes, dict):
                raise TypeError(
                    'labels must be a dict! keys: sample ID or any unique identifier')

        if not len(data) == len(labels) == len(classes):
            raise ValueError('Lengths of data, labels and classes do not match!')
        if not set(list(data)) == set(list(labels)) == set(list(classes)):
            raise ValueError(
                'data, classes and labels dictionaries must have the same keys!')

        num_features_in_elements = np.unique([sample.size for sample in data.values()])
        if len(num_features_in_elements) > 1:
            raise ValueError(
                'different samples have different number of features - invalid!')

        return True