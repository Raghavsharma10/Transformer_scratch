def feature_names(self, names):
        "Stores the text labels for features"

        if len(names) != self.num_features:
            raise ValueError("Number of names do not match the number of features!")
        if not isinstance(names, (Sequence, np.ndarray, np.generic)):
            raise ValueError("Input is not a sequence. "
                             "Ensure names are in the same order "
                             "and length as features.")

        self.__feature_names = np.array(names)