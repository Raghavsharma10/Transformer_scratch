def blast(self):
        """
        Run BLAST analyses of the subsampled FASTQ reads against the NCBI 16S reference database
        """
        logging.info('BLASTing FASTA files against {} database'.format(self.analysistype))
        for _ in range(self.cpus):
            threads = Thread(target=self.blastthreads, args=())
            threads.setDaemon(True)
            threads.start()
        with progressbar(self.runmetadata.samples) as bar:
            for sample in bar:
                if sample.general.bestassemblyfile != 'NA':
                    # Set the name of the BLAST report
                    sample[self.analysistype].blastreport = os.path.join(
                        sample[self.analysistype].outputdir,
                        '{}_{}_blastresults.csv'.format(sample.name, self.analysistype))
                    # Use the NCBI BLASTn command line wrapper module from BioPython to set the parameters of the search
                    blastn = NcbiblastnCommandline(query=sample[self.analysistype].fasta,
                                                   db=os.path.splitext(sample[self.analysistype].baitfile)[0],
                                                   max_target_seqs=1,
                                                   num_threads=self.threads,
                                                   outfmt="'6 qseqid sseqid positive mismatch gaps evalue "
                                                          "bitscore slen length qstart qend qseq sstart send sseq'",
                                                   out=sample[self.analysistype].blastreport)
                    # Add a string of the command to the metadata object
                    sample[self.analysistype].blastcall = str(blastn)
                    # Add the object and the command to the BLAST queue
                    self.blastqueue.put((sample, blastn))
        self.blastqueue.join()