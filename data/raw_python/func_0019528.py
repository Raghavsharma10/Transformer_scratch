def scan(self, seqs, nreport=100, scan_rc=True, normalize=False):
        """
        scan a set of regions / sequences
        """

        if not self.threshold:
            sys.stderr.write(
                "Using default threshold of 0.95. "
                "This is likely not optimal!\n"
                )
            self.set_threshold(threshold=0.95)

        seqs = as_fasta(seqs, genome=self.genome)
           
        it = self._scan_sequences(seqs.seqs, 
                    nreport, scan_rc)
       
        if normalize:
            if len(self.meanstd) == 0:
                self.set_meanstd()
            mean_std = [self.meanstd.get(m_id) for m_id in self.motif_ids]
            means = [x[0] for x in  mean_std]
            stds = [x[1] for x in  mean_std]


        for result in it:
            if normalize:
                zresult = [] 
                for i,mrow in enumerate(result):
                    mrow = [((x[0] - means[i]) / stds[i], x[1], x[2]) for x in mrow]
                    zresult.append(mrow)
                yield zresult
            else:
                yield result