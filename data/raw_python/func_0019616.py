def run(self, fastafile, params=None, tmp=None):
        """
        Run the tool and predict motifs from a FASTA file.

        Parameters
        ----------
        fastafile : str
            Name of the FASTA input file.

        params : dict, optional
            Optional parameters. For some of the tools required parameters
            are passed using this dictionary.

        tmp : str, optional
            Directory to use for creation of temporary files.
       
        Returns
        -------
        motifs : list of Motif instances
            The predicted motifs.

        stdout : str
            Standard out of the tool.
        
        stderr : str
            Standard error of the tool.
        """
        if not self.is_configured():
            raise ValueError("%s is not configured" % self.name)

        if not self.is_installed():
            raise ValueError("%s is not installed or not correctly configured" % self.name)
        
        self.tmpdir = mkdtemp(prefix="{0}.".format(self.name), dir=tmp)
        fastafile = os.path.abspath(fastafile)
 
        try:
            return self._run_program(self.bin(), fastafile, params)
        except KeyboardInterrupt:
            return ([], "Killed", "Killed")