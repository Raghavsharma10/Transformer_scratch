def read_quality_trim(self):
        """
        Perform quality trim, and toss reads below appropriate thresholds
        """
        logging.info('Quality trim')
        for sample in self.metadata:
            sample.sampled_reads = GenObject()
            sample.sampled_reads.outputdir = os.path.join(sample.outputdir, 'sampled')
            sample.sampled_reads.trimmed_dir = os.path.join(sample.sampled_reads.outputdir, 'qualitytrimmed_reads')
            make_path(sample.sampled_reads.trimmed_dir)
            for depth in self.read_depths:
                # Create the depth GenObject
                setattr(sample.sampled_reads, depth, GenObject())
                # Set the depth and output directory attributes for the depth GenObject
                sample.sampled_reads[depth].depth = depth
                sample.sampled_reads[depth].depth_dir = os.path.join(sample.sampled_reads.outputdir, depth)
                # Create the output directory
                make_path(sample.sampled_reads[depth].depth_dir)
                for read_pair in self.read_lengths:
                    # Create the read_pair GenObject within the depth GenObject
                    setattr(sample.sampled_reads[depth], read_pair, GenObject())
                    # Set and create the output directory
                    sample.sampled_reads[depth][read_pair].outputdir = \
                        os.path.join(sample.sampled_reads[depth].depth_dir, read_pair)
                    make_path(sample.sampled_reads[depth][read_pair].outputdir)
                    # Create both forward_reads and reverse_reads sub-GenObjects
                    sample.sampled_reads[depth][read_pair].forward_reads = GenObject()
                    sample.sampled_reads[depth][read_pair].reverse_reads = GenObject()
                    sample.sampled_reads[depth][read_pair].trimmed_dir = \
                        os.path.join(sample.sampled_reads.trimmed_dir,
                                     read_pair)
                    make_path(sample.sampled_reads[depth][read_pair].trimmed_dir)
                    # Extract the forward and reverse reads lengths from the read_pair variable
                    sample.sampled_reads[depth][read_pair].forward_reads.length, \
                        sample.sampled_reads[depth][read_pair].reverse_reads.length = read_pair.split('_')
                    logging.info('Performing quality trimming on reads from sample {name} at depth {depth} '
                                 'for minimum read length {forward}'
                                 .format(name=sample.name,
                                         depth=depth,
                                         forward=sample.sampled_reads[depth][read_pair].forward_reads.length))
                    # Set the attributes for the trimmed forward and reverse reads to use for subsampling
                    sample.sampled_reads[depth][read_pair].trimmed_forwardfastq = \
                        os.path.join(sample.sampled_reads[depth][read_pair].trimmed_dir,
                                     '{name}_{length}_R1.fastq.gz'
                                     .format(name=sample.name,
                                             length=sample.sampled_reads[depth][read_pair].forward_reads.length))
                    sample.sampled_reads[depth][read_pair].trimmed_reversefastq = \
                        os.path.join(sample.sampled_reads[depth][read_pair].trimmed_dir,
                                     '{name}_{length}_R2.fastq.gz'
                                     .format(name=sample.name,
                                             length=sample.sampled_reads[depth][read_pair].forward_reads.length))
                    # Create the trimmed output directory attribute
                    sample.sampled_reads[depth][read_pair].sampled_trimmed_outputdir \
                        = os.path.join(sample.sampled_reads[depth][read_pair].outputdir,
                                       'sampled_trimmed')
                    # Set the name of the forward trimmed reads - include the depth and read length information
                    # This is set now, as the untrimmed files will be removed, and a check is necessary
                    sample.sampled_reads[depth][read_pair].forward_reads.trimmed_sampled_fastq = \
                        os.path.join(sample.sampled_reads[depth][read_pair].sampled_trimmed_outputdir,
                                     '{name}_sampled_{depth}_{read_pair}_R1.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Reverse reads
                    sample.sampled_reads[depth][read_pair].reverse_reads.trimmed_sampled_fastq = \
                        os.path.join(sample.sampled_reads[depth][read_pair].sampled_trimmed_outputdir,
                                     '{name}_sampled_{depth}_{read_pair}_R2.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Sample if the forward output file does not already exist
                    if not os.path.isfile(sample.sampled_reads[depth][read_pair].trimmed_forwardfastq) and \
                            not os.path.isfile(
                                sample.sampled_reads[depth][read_pair].forward_reads.trimmed_sampled_fastq):
                        out, \
                            err, \
                            sample.sampled_reads[depth][read_pair].sample_cmd = \
                            bbtools.bbduk_trim(forward_in=sample.forward_fastq,
                                               forward_out=sample.sampled_reads[depth][read_pair]
                                               .trimmed_forwardfastq,
                                               reverse_in=sample.reverse_fastq,
                                               reverse_out=sample.sampled_reads[depth][read_pair]
                                               .trimmed_reversefastq,
                                               minlength=sample.sampled_reads[depth][read_pair]
                                               .forward_reads.length,
                                               forcetrimleft=0,
                                               returncmd=True,
                                               **{'ziplevel': '9',
                                                  'Xmx': self.mem})
            # Update the JSON file
            self.write_json(sample)