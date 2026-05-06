def revert(self):
        """Revert the state to the version stored on disc."""
        if self.filepath:
            if path.isfile(self.filepath):
                serialised_file = open(self.filepath, "r")
                try:
                    self.state = json.load(serialised_file)
                except ValueError:
                    print("No JSON information could be read from the persistence file - could be empty: %s" % self.filepath)
                    self.state = {}
                finally:
                    serialised_file.close()
            else:
                print("The persistence file has not yet been created or does not exist, so the state cannot be read from it yet.")
        else:
            print("Filepath to the persistence file is not set. State cannot be read.")
            return False