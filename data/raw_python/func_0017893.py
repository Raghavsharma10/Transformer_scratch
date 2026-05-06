def move(self, dst):
        "Closes then moves the file to dst."
        self.close()
        shutil.move(self.path, dst)