def sync(self):
        """Synchronise and update the stored state to the in-memory state."""
        if self.filepath:
            serialised_file = open(self.filepath, "w")
            json.dump(self.state, serialised_file)
            serialised_file.close()
        else:
            print("Filepath to the persistence file is not set. State cannot be synced to disc.")