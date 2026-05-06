def get_closest_match(self, motifs, dbmotifs=None, match="partial", metric="wic",combine="mean", parallel=True, ncpus=None):
        """Return best match in database for motifs.

        Parameters
        ----------
        motifs : list or str
            Filename of motifs or list of motifs.

        dbmotifs : list or str, optional
            Database motifs, default will be used if not specified.

        match : str, optional

        metric : str, optional

        combine : str, optional

        ncpus : int, optional
            Number of threads to use.

        Returns
        -------
        closest_match : dict
        """

        if dbmotifs is None:
            pwm = self.config.get_default_params()["motif_db"]
            pwmdir = self.config.get_motif_dir()
            dbmotifs = os.path.join(pwmdir, pwm)
       
        motifs = parse_motifs(motifs)
        dbmotifs = parse_motifs(dbmotifs)

        dbmotif_lookup = dict([(m.id, m) for m in dbmotifs])

        scores = self.get_all_scores(motifs, dbmotifs, match, metric, combine, parallel=parallel, ncpus=ncpus)
        for motif in scores:
            scores[motif] = sorted(
                    scores[motif].items(), 
                    key=lambda x:x[1][0]
                    )[-1]
        
        for motif in motifs:
            dbmotif, score = scores[motif.id]
            pval, pos, orient = self.compare_motifs(
                    motif, dbmotif_lookup[dbmotif], match, metric, combine, True)
            
            scores[motif.id] = [dbmotif, (list(score) + [pval])]
        
        return scores