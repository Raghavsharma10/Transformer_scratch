def save(self, file_path):
        """
        Method to save the dataset to disk.

        Parameters
        ----------
        file_path : str
            File path to save the current dataset to

        Raises
        ------
        IOError
            If saving to disk is not successful.

        """

        # TODO need a file format that is flexible and efficient to allow the following:
        #   1) being able to read just meta info without having to load the ENTIRE dataset
        #       i.e. use case: compatibility check with #subjects, ids and their classes
        #   2) random access layout: being able to read features for a single subject!

        try:
            file_path = os.path.abspath(file_path)
            with open(file_path, 'wb') as df:
                # pickle.dump(self, df)
                pickle.dump((self.__data, self.__classes, self.__labels,
                             self.__dtype, self.__description, self.__num_features,
                             self.__feature_names),
                            df)
            return
        except IOError as ioe:
            raise IOError('Unable to save the dataset to file: {}', format(ioe))
        except:
            raise