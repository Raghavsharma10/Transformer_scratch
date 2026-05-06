def delete(self):
        """delete the file from the filesystem."""
        if self.isfile:
            os.remove(self.fn)
        elif self.isdir:
            shutil.rmtree(self.fn)