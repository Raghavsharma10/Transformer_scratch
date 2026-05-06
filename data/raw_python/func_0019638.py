def _run_program(self, bin, fastafile, params=None):
        """
        Run Posmo and predict motifs from a FASTA file.

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
        default_params = {}
        if params is not None: 
            default_params.update(params)
        
        width = params.get("width", 8)
        basename = "posmo_in.fa"

        new_file = os.path.join(self.tmpdir, basename)
        shutil.copy(fastafile, new_file)
        
        fastafile = new_file
        #pwmfile = fastafile + ".pwm"
    
        motifs = []
        current_path = os.getcwd()
        os.chdir(self.tmpdir)    
        for n_ones in range(4, min(width, 11), 2):
            x = "1" * n_ones
            outfile = "%s.%s.out" % (fastafile, x)
            cmd = "%s 5000 %s %s 1.6 2.5 %s 200" % (bin, x, fastafile, width)
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) 
            stdout, stderr = p.communicate()
            stdout = stdout.decode()
            stderr = stderr.decode()

            context_file = fastafile.replace(basename, "context.%s.%s.txt" % (basename, x))
            cmd = "%s %s %s simi.txt 0.88 10 2 10" % (bin.replace("posmo","clusterwd"), context_file, outfile)
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE) 
            out, err = p.communicate()
            stdout += out.decode()
            stderr += err.decode()
        
            if os.path.exists(outfile):
                with open(outfile) as f:
                    motifs += self.parse(f, width, n_ones)
        
        os.chdir(current_path)
        
        return motifs, stdout, stderr