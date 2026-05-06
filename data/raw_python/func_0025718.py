def persist(self):
        """a private method that persists an estimator object to the filesystem"""
        if self.object_hash:
            data = dill.dumps(self.object_property)
            f = ContentFile(data)
            self.object_file.save(self.object_hash, f, save=False)
            f.close()
            self._persisted = True
        return self._persisted