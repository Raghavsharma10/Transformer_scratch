def get_checkpoint_files(self):
    """
    Return a list of checkpoint files for this DAG node and its job.
    """
    checkpoint_files = list(self.__checkpoint_files)
    if isinstance(self.job(), CondorDAGJob):
        checkpoint_files = checkpoint_files + self.job().get_checkpoint_files()
    return checkpoint_files