def set_threshold(self, fpr=None, threshold=None):
        """Set motif scanning threshold based on background sequences.

        Parameters
        ----------
        fpr : float, optional
            Desired FPR, between 0.0 and 1.0.

        threshold : float or str, optional
            Desired motif threshold, expressed as the fraction of the 
            difference between minimum and maximum score of the PWM.
            Should either be a float between 0.0 and 1.0 or a filename
            with thresholds as created by 'gimme threshold'.

        """
        if threshold and fpr:
            raise ValueError("Need either fpr or threshold.")
    
        if fpr:
            fpr = float(fpr)
            if not (0.0 < fpr < 1.0):
                raise ValueError("Parameter fpr should be between 0 and 1")
       
        if not self.motifs:
            raise ValueError("please run set_motifs() first")

        thresholds = {}
        motifs = read_motifs(self.motifs)
        
        if threshold is not None:
            self.threshold = parse_threshold_values(self.motifs, threshold) 
            return
        
        if not self.background:
            try:
                self.set_background()
            except:
                raise ValueError("please run set_background() first")
        
        seqs = self.background.seqs

        with Cache(CACHE_DIR) as cache:
            scan_motifs = []
            for motif in motifs:
                k = "{}|{}|{:.4f}".format(motif.hash(), self.background_hash, fpr)
           
                threshold = cache.get(k)
                if threshold is None:
                    scan_motifs.append(motif)
                else:
                    if np.isclose(threshold, motif.pwm_max_score()):
                        thresholds[motif.id] = None 
                    elif np.isclose(threshold, motif.pwm_min_score()):
                        thresholds[motif.id] = 0.0
                    else:
                        thresholds[motif.id] = threshold
                    
            if len(scan_motifs) > 0:
                logger.info("Determining FPR-based threshold")
                for motif, threshold in self._threshold_from_seqs(scan_motifs, seqs, fpr):
                    k = "{}|{}|{:.4f}".format(motif.hash(), self.background_hash, fpr)
                    cache.set(k, threshold)
                    if np.isclose(threshold, motif.pwm_max_score()):
                        thresholds[motif.id] = None
                    elif np.isclose(threshold, motif.pwm_min_score()):
                        thresholds[motif.id] = 0.0
                    else:
                        thresholds[motif.id] = threshold
        self.threshold_str = "{}_{}_{}".format(fpr, threshold, self.background_hash)
        self.threshold = thresholds