def add_checkpoint_file(self, filename):
    """
    Add filename as a checkpoint file for this DAG job.
    """
    if filename not in self.__checkpoint_files:
        self.__checkpoint_files.append(filename)