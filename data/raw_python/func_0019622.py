def _run_program(self, bin, fastafile, params=None):
        """
        Run HMS and predict motifs from a FASTA file.

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
        
        default_params = {"width":10}
        if params is not None: 
            default_params.update(params)
        
        fgfile, summitfile, outfile = self._prepare_files(fastafile)
                
        current_path = os.getcwd()
        os.chdir(self.tmpdir)
        
        cmd = "{} -i {} -w {} -dna 4 -iteration 50 -chain 20 -seqprop -0.1 -strand 2 -peaklocation {} -t_dof 3 -dep 2".format(
                bin, 
                fgfile, 
                params['width'], 
                summitfile)

        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) 
        stdout,stderr = p.communicate()
        
        os.chdir(current_path)
        
        motifs = []
        if os.path.exists(outfile):
            with open(outfile) as f: 
                motifs = self.parse(f)
                for i,m in enumerate(motifs):
                    m.id = "HMS_w{}_{}".format(params['width'], i + 1)
        
        return motifs, stdout, stderr