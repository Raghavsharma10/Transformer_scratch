def fasta(self):
        """
        Convert the subsampled reads to FASTA format using reformat.sh
        """
        logging.info('Converting FASTQ files to FASTA format')
        # Create the threads for the analysis
        for _ in range(self.cpus):
            threads = Thread(target=self.fastathreads, args=())
            threads.setDaemon(True)
            threads.start()
        with progressbar(self.runmetadata.samples) as bar:
            for sample in bar:
                if sample.general.bestassemblyfile != 'NA':
                    # Set the name as the FASTA file - the same as the FASTQ, but with .fa file extension
                    sample[self.analysistype].fasta = \
                        os.path.splitext(sample[self.analysistype].subsampledfastq)[0] + '.fa'
                    # Set the system call
                    sample[self.analysistype].reformatcall = 'reformat.sh in={fastq} out={fasta}'\
                        .format(fastq=sample[self.analysistype].subsampledfastq,
                                fasta=sample[self.analysistype].fasta)
                    # Add the sample to the queue
                    self.fastaqueue.put(sample)
        self.fastaqueue.join()