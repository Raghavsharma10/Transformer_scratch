def on_close(self, filename):
        """Move this file to destination folder."""
        shutil.move(filename, self.destination_folder)
        path, fn = os.path.split(filename)
        return os.path.join(self.destination_folder, fn)