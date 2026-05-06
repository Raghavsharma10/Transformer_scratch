def _run_program(self, bin, fastafile, params=None):
        """
        Run MotifSampler and predict motifs from a FASTA file.

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
        # TODO: test organism
        #cmd = "%s -f %s -b %s -m %s -w %s -n %s -o %s -s %s > /dev/null 2>&1" % (
        cmd = "%s -f %s -b %s -m %s -w %s -n %s -o %s -s %s" % (
                bin, 
                fastafile, 
                params["background_model"], 
                params["pwmfile"], 
                params["width"], 
                params["number"], 
                params["outfile"],
                params["strand"],
                )
        #print cmd
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) 
        stdout, stderr = p.communicate()
        
        #stdout,stderr = "",""
        #p = Popen(cmd, shell=True)
        #p.wait()

        motifs = []
        if os.path.exists(params["outfile"]):
            with open(params["outfile"]) as f:
                motifs = self.parse_out(f)
        
        for motif in motifs:
            motif.id = "%s_%s" % (self.name, motif.id)
        
        return motifs, stdout, stderr