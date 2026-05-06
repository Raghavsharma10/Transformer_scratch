def main(args):
    """Main program"""
    (ribo_file, rna_file, transcript_name, transcriptome_fasta, read_lengths,
     read_offsets, output_path, html_file) = (
         args.ribo_file, args.rna_file, args.transcript_name, args.transcriptome_fasta,
         args.read_lengths, args.read_offsets, args.output_path, args.html_file)

    # error messages (simple format) are written to html file
    fh = logging.FileHandler(html_file)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(ErrorLogFormatter('%(message)s'))
    log.addHandler(fh)

    log.debug('Supplied arguments\n{}'.format(
        '\n'.join(['{:<20}: {}'.format(k, v) for k, v in vars(args).items()])))
    log.debug('Testing debugggg')
    log.info('Checking if required arguments are valid...')
    ribocore.check_required_arguments(
        ribo_file=ribo_file, transcriptome_fasta=transcriptome_fasta, transcript_name=transcript_name)
    log.info('Done')

    if rna_file:
        log.info('Checking if RNA-Seq file is valid...')
        ribocore.check_rna_file(rna_file=rna_file)
        log.info('Done')

    log.info('Checking read lengths...')
    ribocore.check_read_lengths(ribo_file=ribo_file, read_lengths=read_lengths)
    log.info('Done')

    log.info('Checking read offsets...')
    ribocore.check_read_offsets(read_offsets=read_offsets)
    log.info('Done')

    log.info('Checking if each read length has a corresponding offset')
    ribocore.check_read_lengths_offsets(read_lengths=read_lengths, read_offsets=read_offsets)
    log.info('Done')

    log.info('Get sequence and length of the given transcript from FASTA file...')
    record = ribocore.get_fasta_record(transcriptome_fasta, transcript_name)
    transcript_sequence = record[transcript_name]
    transcript_length = len(transcript_sequence)

    log.info('Get ribo-seq read counts and total reads in Ribo-Seq...')
    with ribocore.open_pysam_file(fname=ribo_file, ftype='bam') as bam_fileobj:
        ribo_counts, total_reads = ribocore.get_ribo_counts(
            ribo_fileobj=bam_fileobj, transcript_name=transcript_name,
            read_lengths=read_lengths, read_offsets=read_offsets)

    if not ribo_counts:
        msg = ('No RiboSeq read counts for transcript {}. No plot will be '
               'generated!'.format(transcript_name))
        log.error(msg)
        raise ribocore.RiboPlotError(msg)
    else:
        log.info('Get RNA counts for the given transcript...')
        mrna_counts = {}
        if rna_file:
            try:
                mrna_counts = get_rna_counts(rna_file, transcript_name)
            except OSError as e:
                log.error(e)
                raise

            if not mrna_counts:
                log.warn('No RNA counts for this transcript from the given RNA Seq file. '
                         'RNA-Seq coverage will not be generated')
        else:
            log.debug('No RNA-Seq data provided. Not generating coverage')

        log.info('Get start/stop positions in transcript sequence (3 frames)...')
        codon_positions = get_start_stops(transcript_sequence)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        log.info('Writing RiboSeq read counts for {}'.format(transcript_name))
        with open(os.path.join(output_path, 'RiboCounts.csv'), 'w') as f:
            f.write('"Position","Nucleotide","Frame 1","Frame 2","Frame 3"\n')

            for pos in range(1, transcript_length + 1):
                if pos in ribo_counts:
                    f.write('{0},{1},{2},{3},{4}\n'.format(
                        pos, transcript_sequence[pos - 1], ribo_counts[pos][1], ribo_counts[pos][2], ribo_counts[pos][3]))
                else:
                    f.write('{0},{1},{2},{3},{4}\n'.format(pos, transcript_sequence[pos - 1], 0, 0, 0))

        log.info('Generating RiboPlot...')
        plot_profile(ribo_counts, transcript_name, transcript_length,
                     codon_positions, read_lengths, read_offsets, mrna_counts,
                     color_scheme=args.color_scheme,
                     html_file=args.html_file, output_path=args.output_path)
    log.info('Finished!')