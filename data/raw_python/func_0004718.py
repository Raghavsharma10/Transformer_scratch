def read(self, file_path=None):
        """
        Read the contents of a file.
        :param filename: (str) path to a file in the local file system
        :return: (str) contents of the file, or (False) if not found/not file
        """
        if not file_path:
            file_path = self.file_path

        # abort if the file path does not exist
        if not os.path.exists(file_path):
            self.oops("Sorry, but {} does not exist".format(file_path))
            return False

        # abort if the file path is not a file
        if not os.path.isfile(file_path):
            self.oops("Sorry, but {} is not a file".format(file_path))
            return False

        with open(file_path) as handler:
            return handler.read()