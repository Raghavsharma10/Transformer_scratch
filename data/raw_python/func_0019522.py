def set_background(self, fname=None, genome=None, length=200, nseq=10000):
        """Set the background to use for FPR and z-score calculations.

        Background can be specified either as a genome name or as the 
        name of a FASTA file.
        
        Parameters
        ----------
        fname : str, optional
            Name of FASTA file to use as background.

        genome : str, optional
            Name of genome to use to retrieve random sequences.

        length : int, optional
            Length of genomic sequences to retrieve. The default
            is 200.

        nseq : int, optional
            Number of genomic sequences to retrieve.
        """
        length = int(length)

        if genome and fname:
            raise ValueError("Need either genome or filename for background.")

        if fname:
            if not os.path.exists(fname):
                raise IOError("Background file {} does not exist!".format(fname))

            self.background = Fasta(fname)
            self.background_hash = file_checksum(fname)
            return
        
        if not genome:
            if self.genome:
                genome = self.genome
                logger.info("Using default background: genome {} with length {}".format(
                    genome, length))
            else:
                raise ValueError("Need either genome or filename for background.")
        
        
        logger.info("Using background: genome {} with length {}".format(genome, length))
        with Cache(CACHE_DIR) as cache:
            self.background_hash = "{}\{}".format(genome, int(length))
            fa = cache.get(self.background_hash)
            if not fa:
                fa = RandomGenomicFasta(genome, length, nseq)
                cache.set(self.background_hash, fa)
        self.background = fa