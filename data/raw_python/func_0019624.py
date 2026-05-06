def _run_program(self, bin, fastafile, params=None):
        """
        Run AMD and predict motifs from a FASTA file.

        Parameters
        ----------
        bin : str
            Command used to run the tool.
        
        fastafile : str
            Name of the FASTA input file.

        params : dict, optional
            Optional parameters. For some of the tools required parameters
            are passed using this dictionary.

        Returns
        -------
        motifs : list of Motif instances
            The predicted motifs.

        stdout : str
            Standard out of the tool.
        
        stderr : str
            Standard error of the tool.
        """
        params = self._parse_params(params)

        fgfile = os.path.join(self.tmpdir, "AMD.in.fa")
        outfile = fgfile + ".Matrix"    
        shutil.copy(fastafile, fgfile)
        
        current_path = os.getcwd()
        os.chdir(self.tmpdir)
        
        stdout = ""
        stderr = ""
    
        cmd = "%s -F %s -B %s" % (
                bin, 
                fgfile, 
                params["background"],
                )
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) 
        out,err = p.communicate()
        stdout += out.decode()
        stderr += err.decode()
        
        os.chdir(current_path)
        motifs = []
        if os.path.exists(outfile):
            f = open(outfile)
            motifs = self.parse(f)
            f.close()
        
        return motifs, stdout, stderr