def run_pointfinder(self):
        """
        Run PointFinder on the FASTA sequences extracted from the raw reads
        """
        logging.info('Running PointFinder on FASTA files')
        for i in range(len(self.runmetadata.samples)):
            # Start threads
            threads = Thread(target=self.pointfinder_threads, args=())
            # Set the daemon to True - something to do with thread management
            threads.setDaemon(True)
            # Start the threading
            threads.start()
        # PointFinder requires the path to the blastn executable
        blast_path = shutil.which('blastn')
        for sample in self.runmetadata.samples:
            # Ensure that the attribute storing the name of the FASTA file has been created
            if GenObject.isattr(sample[self.analysistype], 'pointfinderfasta'):
                sample[self.analysistype].pointfinder_outputs = os.path.join(sample[self.analysistype].outputdir,
                                                                             'pointfinder_outputs')
                # Don't run the analyses if the outputs have already been created
                if not os.path.isfile(os.path.join(sample[self.analysistype].pointfinder_outputs,
                                                   '{samplename}_blastn_results.tsv'.format(samplename=sample.name))):
                    make_path(sample[self.analysistype].pointfinder_outputs)
                    # Create and run the PointFinder system call
                    pointfinder_cmd = \
                        'python -m pointfinder.PointFinder -i {input} -s {species} -p {db_path} -m blastn ' \
                        '-o {output_dir} -m_p {blast_path}'\
                        .format(input=sample[self.analysistype].pointfinderfasta,
                                species=sample[self.analysistype].pointfindergenus,
                                db_path=self.targetpath,
                                output_dir=sample[self.analysistype].pointfinder_outputs,
                                blast_path=blast_path)
                    self.queue.put(pointfinder_cmd)
        self.queue.join()