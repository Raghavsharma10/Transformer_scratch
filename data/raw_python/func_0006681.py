def simulate_reads(self):
        """
        Use the PacBio assembly FASTA files to generate simulated reads of appropriate forward and reverse lengths
        at different depths of sequencing using randomreads.sh from the bbtools suite
        """
        logging.info('Read simulation')
        for sample in self.metadata:
            # Create the simulated_reads GenObject
            sample.simulated_reads = GenObject()
            # Iterate through all the desired depths of coverage
            for depth in self.read_depths:
                # Create the depth GenObject
                setattr(sample.simulated_reads, depth, GenObject())
                # Set the depth and output directory attributes for the depth GenObject
                sample.simulated_reads[depth].depth = depth
                sample.simulated_reads[depth].depth_dir = os.path.join(sample.outputdir, 'simulated', depth)
                # Create the output directory
                make_path(sample.simulated_reads[depth].depth_dir)
                # Iterate through all the desired forward and reverse read pair lengths
                for read_pair in self.read_lengths:
                    # Create the read_pair GenObject within the depth GenObject
                    setattr(sample.simulated_reads[depth], read_pair, GenObject())
                    # Set and create the output directory
                    sample.simulated_reads[depth][read_pair].outputdir = \
                        os.path.join(sample.simulated_reads[depth].depth_dir, read_pair)
                    make_path(sample.simulated_reads[depth][read_pair].outputdir)
                    # Create both forward_reads and reverse_reads sub-GenObjects
                    sample.simulated_reads[depth][read_pair].forward_reads = GenObject()
                    sample.simulated_reads[depth][read_pair].reverse_reads = GenObject()
                    # Extract the forward and reverse reads lengths from the read_pair variable
                    sample.simulated_reads[depth][read_pair].forward_reads.length, \
                        sample.simulated_reads[depth][read_pair].reverse_reads.length = read_pair.split('_')
                    # Set the name of the forward reads - include the depth and read length information
                    sample.simulated_reads[depth][read_pair].forward_reads.fastq = \
                        os.path.join(sample.simulated_reads[depth][read_pair].outputdir,
                                     '{name}_{depth}_{read_pair}_R1.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Reverse reads
                    sample.simulated_reads[depth][read_pair].reverse_reads.fastq = \
                        os.path.join(sample.simulated_reads[depth][read_pair].outputdir,
                                     '{name}_{depth}_{read_pair}_R2.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Create the trimmed output directory attribute
                    sample.simulated_reads[depth][read_pair].simulated_trimmed_outputdir \
                        = os.path.join(sample.simulated_reads[depth][read_pair].outputdir,
                                       'simulated_trimmed')
                    # Set the name of the forward trimmed reads - include the depth and read length information
                    # This is set now, as the untrimmed files will be removed, and a check is necessary
                    sample.simulated_reads[depth][read_pair].forward_reads.trimmed_simulated_fastq = \
                        os.path.join(sample.simulated_reads[depth][read_pair].simulated_trimmed_outputdir,
                                     '{name}_simulated_{depth}_{read_pair}_R1.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Reverse reads
                    sample.simulated_reads[depth][read_pair].reverse_reads.trimmed_simulated_fastq = \
                        os.path.join(sample.simulated_reads[depth][read_pair].simulated_trimmed_outputdir,
                                     '{name}_simulated_{depth}_{read_pair}_R2.fastq.gz'
                                     .format(name=sample.name,
                                             depth=depth,
                                             read_pair=read_pair))
                    # Calculate the number of reads required for the forward and reverse reads to yield the
                    # desired coverage depth e.g. 5Mbp genome at 20X coverage: 100Mbp in reads. 50bp forward reads
                    # 150bp reverse reads: forward proportion is 50 / (150 + 50) = 0.25 (and reverse is 0.75).
                    # Forward total reads is 25Mbp (75Mbp reverse). Number of reads required = 25Mbp / 50 bp
                    # 500000 reads total (same for reverse, as the reads are longer)
                    sample.simulated_reads[depth][read_pair].num_reads = \
                        int(sample.assembly_length *
                            int(depth) *
                            (int(sample.simulated_reads[depth][read_pair].forward_reads.length) /
                             (int(sample.simulated_reads[depth][read_pair].forward_reads.length) +
                              int(sample.simulated_reads[depth][read_pair].reverse_reads.length)
                              )
                             ) /
                            int(sample.simulated_reads[depth][read_pair].forward_reads.length)
                            )
                    logging.info(
                        'Simulating {num_reads} paired reads for sample {name} with the following parameters:\n'
                        'depth {dp}, forward reads {fl}bp, and reverse reads {rl}bp'
                        .format(num_reads=sample.simulated_reads[depth][read_pair].num_reads,
                                dp=depth,
                                name=sample.name,
                                fl=sample.simulated_reads[depth][read_pair].forward_reads.length,
                                rl=sample.simulated_reads[depth][read_pair].reverse_reads.length))
                    # If the reverse reads are set to 0, supply different parameters to randomreads
                    if sample.simulated_reads[depth][read_pair].reverse_reads.length != '0':
                        # Ensure that both the simulated reads, and the trimmed simulated reads files don't
                        # exist before simulating the reads
                        if not os.path.isfile(sample.simulated_reads[depth][read_pair].forward_reads.fastq) and \
                                not os.path.isfile(
                                    sample.simulated_reads[depth][read_pair].forward_reads.trimmed_simulated_fastq):
                            # Use the randomreads method in the OLCTools bbtools wrapper to simulate the reads
                            out, \
                                err, \
                                sample.simulated_reads[depth][read_pair].forward_reads.simulate_call = bbtools\
                                .randomreads(reference=sample.bestassemblyfile,
                                             length=sample.simulated_reads[depth][read_pair].reverse_reads.length,
                                             reads=sample.simulated_reads[depth][read_pair].num_reads,
                                             out_fastq=sample.simulated_reads[depth][read_pair].forward_reads.fastq,
                                             paired=True,
                                             returncmd=True,
                                             **{'ziplevel': '9',
                                                'illuminanames': 't',
                                                'Xmx': self.mem}
                                             )
                        else:
                            try:
                                forward_size = os.path.getsize(sample.simulated_reads[depth][read_pair]
                                                               .forward_reads.fastq)
                            except FileNotFoundError:
                                forward_size = 0
                            try:
                                reverse_size = os.path.getsize(sample.simulated_reads[depth][read_pair]
                                                               .reverse_reads.fastq)
                            except FileNotFoundError:
                                reverse_size = 0
                            if forward_size <= 100 or reverse_size <= 100:
                                try:
                                    os.remove(sample.simulated_reads[depth][read_pair].forward_reads.fastq)
                                except FileNotFoundError:
                                    pass
                                try:
                                    os.remove(sample.simulated_reads[depth][read_pair].reverse_reads.fastq)
                                except FileNotFoundError:
                                    pass
                                # Use the randomreads method in the OLCTools bbtools wrapper to simulate the reads
                                out, \
                                err, \
                                sample.simulated_reads[depth][read_pair].forward_reads.simulate_call = bbtools \
                                    .randomreads(reference=sample.bestassemblyfile,
                                                 length=sample.simulated_reads[depth][read_pair].reverse_reads.length,
                                                 reads=sample.simulated_reads[depth][read_pair].num_reads,
                                                 out_fastq=sample.simulated_reads[depth][read_pair].forward_reads.fastq,
                                                 paired=True,
                                                 returncmd=True,
                                                 **{'ziplevel': '9',
                                                    'illuminanames': 't'}
                                                 )
                    else:
                        if not os.path.isfile(sample.simulated_reads[depth][read_pair].forward_reads.fastq):
                            # Use the randomreads method in the OLCTools bbtools wrapper to simulate the reads
                            out, \
                                err, \
                                sample.simulated_reads[depth][read_pair].forward_reads.simulate_call = bbtools\
                                .randomreads(reference=sample.bestassemblyfile,
                                             length=sample.simulated_reads[depth][read_pair].forward_reads.length,
                                             reads=sample.simulated_reads[depth][read_pair].num_reads,
                                             out_fastq=sample.simulated_reads[depth][read_pair].forward_reads.fastq,
                                             paired=False,
                                             returncmd=True,
                                             **{'ziplevel': '9',
                                                'illuminanames': 't'}
                                             )
                # Update the JSON file
                self.write_json(sample)