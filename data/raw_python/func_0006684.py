def sample_reads(self):
        """
        For each PacBio assembly, sample reads from corresponding FASTQ files for appropriate forward and reverse
        lengths and sequencing depths using reformat.sh from the bbtools suite
        """
        logging.info('Read sampling')
        for sample in self.metadata:
            # Iterate through all the desired depths of coverage
            for depth in self.read_depths:
                for read_pair in self.read_lengths:
                    # Set the name of the output directory
                    sample.sampled_reads[depth][read_pair].sampled_outputdir \
                        = os.path.join(sample.sampled_reads[depth][read_pair].outputdir, 'sampled')
                    # Set the name of the forward reads - include the depth and read length information
                    sample.sampled_reads[depth][read_pair].forward_reads.fastq = \
                        os.path.join(sample.sampled_reads[depth][read_pair].sampled_outputdir,
                                     '{name}_{depth}_{read_pair}_R1.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Reverse reads
                    sample.sampled_reads[depth][read_pair].reverse_reads.fastq = \
                        os.path.join(sample.sampled_reads[depth][read_pair].sampled_outputdir,
                                     '{name}_{depth}_{read_pair}_R2.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    logging.info(
                        'Sampling {num_reads} paired reads for sample {name} with the following parameters:\n'
                        'depth {dp}, forward reads {fl}bp, and reverse reads {rl}bp'
                        .format(num_reads=sample.simulated_reads[depth][read_pair].num_reads,
                                dp=depth,
                                name=sample.name,
                                fl=sample.sampled_reads[depth][read_pair].forward_reads.length,
                                rl=sample.sampled_reads[depth][read_pair].reverse_reads.length))
                    # Use the reformat method in the OLCTools bbtools wrapper
                    # Note that upsample=t is used to ensure that the target number of reads (samplereadstarget) is met
                    if not os.path.isfile(sample.sampled_reads[depth][read_pair].forward_reads.trimmed_sampled_fastq):
                        out, \
                            err, \
                            sample.sampled_reads[depth][read_pair].sample_call = bbtools \
                            .reformat_reads(forward_in=sample.sampled_reads[depth][read_pair].trimmed_forwardfastq,
                                            reverse_in=sample.sampled_reads[depth][read_pair].trimmed_reversefastq,
                                            forward_out=sample.sampled_reads[depth][read_pair].forward_reads.fastq,
                                            reverse_out=sample.sampled_reads[depth][read_pair].reverse_reads.fastq,
                                            returncmd=True,
                                            **{'samplereadstarget': sample.simulated_reads[depth][read_pair].num_reads,
                                               'upsample': 't',
                                               'minlength':
                                                   sample.sampled_reads[depth][read_pair].forward_reads.length,
                                               'ziplevel': '9',
                                               'tossbrokenreads': 't',
                                               'tossjunk': 't',
                                               'Xmx': self.mem
                                               }
                                            )
                    # # Remove the trimmed reads, as they are no longer necessary
                    # try:
                    #     os.remove(sample.sampled_reads[depth][read_pair].trimmed_forwardfastq)
                    #     os.remove(sample.sampled_reads[depth][read_pair].trimmed_reversefastq)
                    # except FileNotFoundError:
                    #     pass
            # Update the JSON file
            self.write_json(sample)