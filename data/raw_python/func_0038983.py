def __load(self, path):
        """Method to load the serialized dataset from disk."""
        try:
            path = os.path.abspath(path)
            with open(path, 'rb') as df:
                # loaded_dataset = pickle.load(df)
                self.__data, self.__classes, self.__labels, \
                self.__dtype, self.__description, \
                self.__num_features, self.__feature_names = pickle.load(df)

            # ensure the loaded dataset is valid
            self.__validate(self.__data, self.__classes, self.__labels)

        except IOError as ioe:
            raise IOError('Unable to read the dataset from file: {}', format(ioe))
        except:
            raise