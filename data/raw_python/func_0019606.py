def create_index(self,fasta_dir=None, index_dir=None):
        """Index all fasta-files in fasta_dir (one sequence per file!) and
        store the results in index_dir"""
        
        # Use default directories if they are not supplied
        if not fasta_dir:
            fasta_dir = self.fasta_dir

        if not index_dir:
            index_dir = self.index_dir

        # Can't continue if we still don't have an index_dir or fasta_dir
        if not fasta_dir:
            print("fasta_dir not defined!")
            sys.exit(1)
        
        if not index_dir:
            print("index_dir not defined!")
            sys.exit(1)
        
        index_dir = os.path.abspath(index_dir)
        fasta_dir = os.path.abspath(fasta_dir)

        self.index_dir = index_dir

        # Prepare index directory
        if not os.path.exists(index_dir):
            try:
                os.mkdir(index_dir)
            except OSError as e:
                if e.args[0] == 13:
                    sys.stderr.write("No permission to create index directory. Superuser access needed?\n")
                    sys.exit()
                else:
                    sys.stderr.write(e)

        # Directories need to exist
        self._check_dir(fasta_dir)
        self._check_dir(index_dir)

        # Get all fasta-files 

        fastafiles = find_by_ext(fasta_dir, FASTA_EXT)
        if not(fastafiles):
            msg = "No fastafiles found in {} with extension in {}".format(
                                        fasta_dir, ",".join(FASTA_EXT))
            raise IOError(msg)

        # param_file will hold all the information about the location of the fasta-files, indeces and 
        # length of the sequences
        param_file = os.path.join(index_dir, self.param_file)
        size_file = os.path.join(index_dir, self.size_file)
        
        try:
            out = open(param_file, "w")
        except IOError as e:
            if e.args[0] == 13:
                sys.stderr.write("No permission to create files in index directory. Superuser access needed?\n")
                sys.exit()
            else:
                sys.stderr.write(e)
        s_out = open(size_file, "w")

        for fasta_file in fastafiles:
            #sys.stderr.write("Indexing %s\n" % fasta_file)
            f = open(fasta_file)
            line = f.readline()
            if not line.startswith(">"):
                sys.stderr.write("%s is not a valid FASTA file, expected > at first line\n" % fasta_file)
                sys.exit()
            
            seqname = line.strip().replace(">", "")
            line = f.readline()
            line_size = len(line.strip())

            total_size = 0 
            while line:
                line = line.strip()
                if line.startswith(">"):
                    sys.stderr.write("Sorry, can only index genomes with "
                    "one sequence per FASTA file\n%s contains multiple "
                    "sequences\n" % fasta_file)
                    sys.exit()
                
                total_size += len(line)
                line = f.readline()

            index_file = os.path.join(index_dir, "%s.index" % seqname)

            out.write("{}\t{}\t{}\t{}\t{}\n".format(
                seqname, fasta_file, index_file, line_size, total_size))
            s_out.write("{}\t{}\n".format(seqname, total_size))
            
            self._make_index(fasta_file, index_file)
            f.close()
        out.close()
        s_out.close()

        # Read the index we just made so we can immediately use it
        self._read_index_file()