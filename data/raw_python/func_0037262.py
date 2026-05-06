def write(self):
        """ Encrypts and writes the current state back onto the filesystem """
        with open(self.filepath, 'wb') as outfile:
            outfile.write(
                self.fernet.encrypt(
                    yaml.dump(self.data, encoding='utf-8')))