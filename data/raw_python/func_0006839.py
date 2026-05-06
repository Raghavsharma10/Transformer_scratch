def subsample(self):
        """
        Subsample 1000 reads from the baited files
        """
        # Create the threads for the analysis
        logging.info('Subsampling FASTQ reads')
        for _ in range(self.cpus):
            threads = Thread(target=self.subsamplethreads, args=())
            threads.setDaemon(True)
            threads.start()
        with progressbar(self.runmetadata.samples) as bar:
            for sample in bar:
                if sample.general.bestassemblyfile != 'NA':
                    # Set the name of the subsampled FASTQ file
                    sample[self.analysistype].subsampledfastq = \
                        os.path.splitext(sample[self.analysistype].baitedfastq)[0] + '_subsampled.fastq'
                    # Set the system call
                    sample[self.analysistype].seqtkcall = 'reformat.sh in={} out={} samplereadstarget=1000'\
                        .format(sample[self.analysistype].baitedfastq,
                                sample[self.analysistype].subsampledfastq)
                    # Add the sample to the queue
                    self.samplequeue.put(sample)
        self.samplequeue.join()