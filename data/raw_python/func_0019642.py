def _run_program(self, bin, fastafile, params=None):
        """
        Run MEME and predict motifs from a FASTA file.

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
        default_params = {"width":10, "single":False, "number":10}
        if params is not None: 
            default_params.update(params)
        
        tmp = NamedTemporaryFile(dir=self.tmpdir)
        tmpname = tmp.name
    
        strand = "-revcomp"
        width = default_params["width"]
        number = default_params["number"]
        
        cmd = [bin, fastafile, "-text","-dna","-nostatus","-mod", "zoops","-nmotifs", "%s" % number, "-w","%s" % width, "-maxsize", "10000000"]
        if not default_params["single"]:
            cmd.append(strand)
        
        #sys.stderr.write(" ".join(cmd) + "\n")
        p = Popen(cmd, bufsize=1, stderr=PIPE, stdout=PIPE) 
        stdout,stderr = p.communicate()

        motifs = []
        motifs = self.parse(io.StringIO(stdout.decode()))
        
        # Delete temporary files
        tmp.close()
         
        return motifs, stdout, stderr