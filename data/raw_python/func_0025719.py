def load(self):
        """a private method that loads an estimator object from the filesystem"""
        if self.is_file_persisted:
            self.object_file.open()
            temp = dill.loads(self.object_file.read())
            self.set_object(temp)
            self.object_file.close()