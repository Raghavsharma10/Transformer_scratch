def main(sequencepath, report, refseq_database, num_threads=12, start=time.time()):
    """
    Run the appropriate functions in order
    :param sequencepath: path of folder containing FASTA genomes
    :param report: boolean to determine whether a report is to be created
    :param refseq_database: Path to reduced refseq database sketch
    :param num_threads: Number of threads to run mash/other stuff on
    :return: gc_dict, contig_dist_dict, longest_contig_dict, genome_length_dict, num_contigs_dict, n50_dict, n75_dict, \
        n90_dict, l50_dict, l75_dict, l90_dict, orf_dist_dict
    """
    files = find_files(sequencepath)
    file_dict = filer(files)
    printtime('Using MASH to determine genera of samples', start)
    genus_dict = find_genus(file_dict, refseq_database, threads=num_threads)
    file_records = fasta_records(file_dict)
    printtime('Collecting basic quality metrics', start)
    contig_len_dict, gc_dict = fasta_stats(file_dict, file_records)
    contig_dist_dict = find_contig_distribution(contig_len_dict)
    longest_contig_dict = find_largest_contig(contig_len_dict)
    genome_length_dict = find_genome_length(contig_len_dict)
    num_contigs_dict = find_num_contigs(contig_len_dict)
    n50_dict = find_n50(contig_len_dict, genome_length_dict)
    n75_dict = find_n75(contig_len_dict, genome_length_dict)
    n90_dict = find_n90(contig_len_dict, genome_length_dict)
    l50_dict = find_l50(contig_len_dict, genome_length_dict)
    l75_dict = find_l75(contig_len_dict, genome_length_dict)
    l90_dict = find_l90(contig_len_dict, genome_length_dict)
    printtime('Using prodigal to calculate number of ORFs in each sample', start)
    orf_file_dict = predict_orfs(file_dict, num_threads=num_threads)
    orf_dist_dict = find_orf_distribution(orf_file_dict)
    if report:
        reporter(gc_dict, contig_dist_dict, longest_contig_dict, genome_length_dict, num_contigs_dict, n50_dict,
                 n75_dict, n90_dict, l50_dict, l75_dict, l90_dict, orf_dist_dict, genus_dict, sequencepath)
    printtime('Features extracted!', start)
    return gc_dict, contig_dist_dict, longest_contig_dict, genome_length_dict, num_contigs_dict, n50_dict, n75_dict, \
        n90_dict, l50_dict, l75_dict, l90_dict, orf_dist_dict