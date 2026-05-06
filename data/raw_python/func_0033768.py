def add_checkpoint_file(self,filename):
    """
    Add filename as a checkpoint file for this DAG node
    @param filename: checkpoint filename to add
    """
    if filename not in self.__checkpoint_files:
        self.__checkpoint_files.append(filename)
        if not isinstance(self.job(), CondorDAGManJob):
            if self.job().get_universe() == 'grid':
                self.add_checkpoint_macro(filename)