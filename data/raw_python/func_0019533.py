def get_all_scores(self, motifs, dbmotifs, match, metric, combine, 
                            pval=False, parallel=True, trim=None, ncpus=None):
        """Pairwise comparison of a set of motifs compared to reference motifs.

        Parameters
        ----------
        motifs : list
            List of Motif instances.

        dbmotifs : list
            List of Motif instances.

        match : str
            Match can be "partial", "subtotal" or "total". Not all metrics use 
            this.

        metric : str
            Distance metric.

        combine : str
            Combine positional scores using "mean" or "sum". Not all metrics
            use this.

        pval : bool , optional
            Calculate p-vale of match.
        
        parallel : bool , optional
            Use multiprocessing for parallel execution. True by default.

        trim : float or None
            If a float value is specified, motifs are trimmed used this IC 
            cutoff before comparison.

        ncpus : int or None
            Specifies the number of cores to use for parallel execution.

        Returns
        -------
        scores : dict
            Dictionary with scores.
        """
        # trim motifs first, if specified
        if trim:
            for m in motifs:
                m.trim(trim)
            for m in dbmotifs:
                m.trim(trim)
        
        # hash of result scores
        scores = {}
        
        if parallel:    
            # Divide the job into big chunks, to keep parallel overhead to minimum
            # Number of chunks = number of processors available
            if ncpus is None:
                ncpus = int(MotifConfig().get_default_params()["ncpus"])

            pool = Pool(processes=ncpus, maxtasksperchild=1000)
 
            batch_len = len(dbmotifs) // ncpus
            if batch_len <= 0:
                batch_len = 1
            jobs = []
            for i in range(0, len(dbmotifs), batch_len): 
                # submit jobs to the job server
                
                p = pool.apply_async(_get_all_scores, 
                    args=(self, motifs, dbmotifs[i: i + batch_len], match, metric, combine, pval))
                jobs.append(p)
            
            pool.close()
            for job in jobs:
                # Get the job result
                result = job.get()
                # and update the result score
                for m1,v in result.items():
                    for m2, s in v.items():
                        if m1 not in scores:
                            scores[m1] = {}
                        scores[m1][m2] = s
        
            pool.join()
        else:
            # Do the whole thing at once if we don't want parallel
            scores = _get_all_scores(self, motifs, dbmotifs, match, metric, combine, pval)
        
        return scores