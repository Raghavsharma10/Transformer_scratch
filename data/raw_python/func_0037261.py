def read(self):
        """ Reads and decrypts data from the filesystem """
        if path.exists(self.filepath):
            with open(self.filepath, 'rb') as infile:
                self.data = yaml.load(
                    self.fernet.decrypt(infile.read()))
        else:
            self.data = dict()