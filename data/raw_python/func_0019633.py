def _run_program(self, bin, fastafile, params=None):
        """
        Run MDmodule and predict motifs from a FASTA file.

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
        default_params = {"width":10, "number":10}
        if params is not None: 
            default_params.update(params)
        
        new_file = os.path.join(self.tmpdir, "mdmodule_in.fa")
        shutil.copy(fastafile, new_file)
        
        fastafile = new_file
        pwmfile = fastafile + ".out"
    
        width = default_params['width']
        number = default_params['number']
    
        current_path = os.getcwd()
        os.chdir(self.tmpdir)    
        cmd = "%s -i %s -a 1 -o %s -w %s -t 100 -r %s" % (bin, fastafile, pwmfile, width, number)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) 
        stdout,stderr = p.communicate()
        
        stdout = "cmd: {}\n".format(cmd) + stdout.decode() 
            
        motifs = []
        if os.path.exists(pwmfile):
            with open(pwmfile) as f:
                motifs = self.parse(f)
        
        os.chdir(current_path)
        
        for motif in motifs:
            motif.id = "%s_%s" % (self.name, motif.id)
        
        return motifs, stdout, stderr