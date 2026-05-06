def write_id3(self, filename):
        """ Write id3 tags """
        if not os.path.exists(filename):
            raise ValueError("File doesn't exists.")

        self.mapper.write(filename)