def backup(self):
        """Backups files with the same name of the instance filename"""
        count = 0
        name = "{}.bkp".format(self.filename)
        backup = os.path.join(self.cwd, name)
        while os.path.exists(backup):
            count += 1
            name = "{}.bkp{}".format(self.filename, count)
            backup = os.path.join(self.cwd, name)
        self.hey("Moving existing {} to {}".format(self.filename, name))
        os.rename(os.path.join(self.cwd, self.filename), backup)