def add_motifs(self, args):
        """Add motifs to the result object."""
        self.lock.acquire()
        # Callback function for motif programs
        if args is None or len(args) != 2 or len(args[1]) != 3:
            try:
                job = args[0]
                logger.warn("job %s failed", job)
                self.finished.append(job)
            except Exception:
                logger.warn("job failed") 
            return
        
        job, (motifs, stdout, stderr) = args

        logger.info("%s finished, found %s motifs", job, len(motifs))
        
        for motif in motifs:
            if self.do_counter:
                self.counter += 1    
                motif.id = "gimme_{}_".format(self.counter) + motif.id
            f = open(self.outfile, "a")
            f.write("%s\n" % motif.to_pfm())
            f.close()
            self.motifs.append(motif)
            
        if self.do_stats and len(motifs) > 0:
            #job_id = "%s_%s" % (motif.id, motif.to_consensus())
            logger.debug("Starting stats job of %s motifs", len(motifs))
            for bg_name, bg_fa in self.background.items():
                job = self.job_server.apply_async(
                                    mp_calc_stats, 
                                    (motifs, self.fg_fa, bg_fa, bg_name), 
                                    callback=self.add_stats
                                    )
                self.stat_jobs.append(job)
        
        logger.debug("stdout %s: %s", job, stdout)
        logger.debug("stdout %s: %s", job, stderr)
        self.finished.append(job)
        self.lock.release()