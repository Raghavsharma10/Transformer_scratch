def __copy(self, other):
        """Copy constructor."""
        self.__data = copy.deepcopy(other.data)
        self.__classes = copy.deepcopy(other.classes)
        self.__labels = copy.deepcopy(other.labels)
        self.__dtype = copy.deepcopy(other.dtype)
        self.__description = copy.deepcopy(other.description)
        self.__feature_names = copy.deepcopy(other.feature_names)
        self.__num_features = copy.deepcopy(other.num_features)

        return self