def read_length_adjust(self, analysistype):
        """
        Trim the reads to the correct length using reformat.sh
        :param analysistype: current analysis type. Will be either 'simulated' or 'sampled'
        """
        logging.info('Trimming {at} reads'.format(at=analysistype))
        for sample in self.metadata:
            # Iterate through all the desired depths of coverage
            for depth in self.read_depths:
                for read_pair in self.read_lengths:
                    # Create variables using the analysis type. These will be used in setting GenObject attributes
                    read_type = '{at}_reads'.format(at=analysistype)
                    fastq_type = 'trimmed_{at}_fastq'.format(at=analysistype)
                    logging.info(
                        'Trimming forward {at} reads for sample {name} at depth {depth} to length {length}'
                        .format(at=analysistype,
                                name=sample.name,
                                depth=depth,
                                length=sample[read_type][depth][read_pair].forward_reads.length))
                    # Create the output path if necessary
                    make_path(os.path.dirname(sample[read_type][depth][read_pair].forward_reads[fastq_type]))
                    if sample[read_type][depth][read_pair].reverse_reads.length != '0':
                        # Use the reformat method in the OLCTools bbtools wrapper to trim the reads
                        out, \
                            err, \
                            sample[read_type][depth][read_pair].forward_reads.sample_call = bbtools \
                            .reformat_reads(forward_in=sample[read_type][depth][read_pair].forward_reads.fastq,
                                            reverse_in=None,
                                            forward_out=sample[read_type][depth][read_pair].forward_reads[fastq_type],
                                            returncmd=True,
                                            **{'ziplevel': '9',
                                               'forcetrimright':
                                                   sample[read_type][depth][read_pair].forward_reads.length,
                                               'tossbrokenreads': 't',
                                               'tossjunk': 't',
                                               'Xmx': self.mem
                                               }
                                            )
                        # # Remove the untrimmed reads
                        # try:
                        #     os.remove(sample[read_type][depth][read_pair].forward_reads.fastq)
                        # except FileNotFoundError:
                        #     pass

                    else:
                        # If the files do not need to be trimmed, create a symlink to the original file
                        relative_symlink(sample[read_type][depth][read_pair].forward_reads.fastq,
                                         os.path.dirname(sample[read_type][depth][read_pair].
                                                         forward_reads[fastq_type]),
                                         os.path.basename(sample[read_type][depth][read_pair].
                                                          forward_reads[fastq_type])
                                         )
                    # Same as above, but for the reverse reads
                    logging.info(
                        'Trimming reverse {at} reads for sample {name} at depth {depth} to length {length}'
                        .format(at=analysistype,
                                name=sample.name,
                                depth=depth,
                                length=sample[read_type][depth][read_pair].reverse_reads.length))
                    if sample[read_type][depth][read_pair].reverse_reads.length != '0':
                        # Use the reformat method in the OLCTools bbtools wrapper to trim the reads
                        out, \
                            err, \
                            sample[read_type][depth][read_pair].reverse_reads.sample_call = bbtools \
                            .reformat_reads(forward_in=sample[read_type][depth][read_pair].reverse_reads.fastq,
                                            reverse_in=None,
                                            forward_out=sample[read_type][depth][read_pair].reverse_reads[fastq_type],
                                            returncmd=True,
                                            **{'ziplevel': '9',
                                               'forcetrimright':
                                                   sample[read_type][depth][read_pair].reverse_reads.length,
                                               'tossbrokenreads': 't',
                                               'tossjunk': 't',
                                               'Xmx': self.mem
                                               })
                        # # Remove the untrimmed reads
                        # try:
                        #     os.remove(sample[read_type][depth][read_pair].reverse_reads.fastq)
                        # except FileNotFoundError:
                        #     pass
            # Update the JSON file
            self.write_json(sample)