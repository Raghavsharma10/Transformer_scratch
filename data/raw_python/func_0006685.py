def link_reads(self, analysistype):
        """
        Create folders with relative symlinks to the desired simulated/sampled reads. These folders will contain all
        the reads created for each sample, and will be processed with GeneSippr and COWBAT pipelines
        :param analysistype: Current analysis type. Will either be 'simulated' or 'sampled'
        """
        logging.info('Linking {at} reads'.format(at=analysistype))
        for sample in self.metadata:
            # Create the output directories
            genesippr_dir = os.path.join(self.path, 'genesippr', sample.name)
            sample.genesippr_dir = genesippr_dir
            make_path(genesippr_dir)
            cowbat_dir = os.path.join(self.path, 'cowbat', sample.name)
            sample.cowbat_dir = cowbat_dir
            make_path(cowbat_dir)
            # Iterate through all the desired depths of coverage
            for depth in self.read_depths:
                for read_pair in self.read_lengths:
                    # Create variables using the analysis type. These will be used in setting GenObject attributes
                    read_type = '{at}_reads'.format(at=analysistype)
                    fastq_type = 'trimmed_{at}_fastq'.format(at=analysistype)
                    # Link reads to both output directories
                    for output_dir in [genesippr_dir, cowbat_dir]:
                        # If the original reads are shorter than the specified read length, the FASTQ files will exist,
                        # but will be empty. Do not create links for these files
                        size = os.path.getsize(sample[read_type][depth][read_pair].forward_reads[fastq_type])
                        if size > 20:
                            # Create relative symlinks to the FASTQ files - use the relative path from the desired
                            # output directory to the read storage path e.g.
                            # ../../2013-SEQ-0072/simulated/40/50_150/simulated_trimmed/2013-SEQ-0072_simulated_40_50_150_R1.fastq.gz
                            # is the relative path to the output_dir. The link name is the base name of the reads
                            # joined to the desired output directory e.g.
                            # output_dir/2013-SEQ-0072/2013-SEQ-0072_simulated_40_50_150_R1.fastq.gz
                            relative_symlink(sample[read_type][depth][read_pair].forward_reads[fastq_type],
                                             output_dir)
                            # Original FASTQ files
                            relative_symlink(sample.forward_fastq,
                                             output_dir)
                            relative_symlink(sample.reverse_fastq,
                                             output_dir)
                        # Reverse reads
                        try:
                            size = os.path.getsize(sample[read_type][depth][read_pair].reverse_reads[fastq_type])
                            if size > 20:
                                relative_symlink(sample[read_type][depth][read_pair].reverse_reads[fastq_type],
                                                 output_dir)
                        except FileNotFoundError:
                            pass