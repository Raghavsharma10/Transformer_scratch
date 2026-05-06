def load(self, file_path):
        ''' Load configuration from a specific file '''
        self.clear()
        self.__config = self.read_file(file_path)