def reporter(gc_dict, contig_dist_dict, longest_contig_dict, genome_length_dict, num_contigs_dict, n50_dict, n75_dict,
             n90_dict, l50_dict, l75_dict, l90_dict, orf_dist_dict, genus_dict, sequencepath):
    """
    Create a report of all the extracted features
    :param gc_dict: dictionary of strain name: GC%
    :param contig_dist_dict: dictionary of strain: tuple of contig distribution frequencies
    :param longest_contig_dict: dictionary of strain name: longest contig
    :param genome_length_dict: dictionary of strain name: total genome length
    :param num_contigs_dict: dictionary of strain name: total number of contigs
    :param n50_dict: dictionary of strain name: N50
    :param n75_dict: dictionary of strain name: N75
    :param n90_dict: dictionary of strain name: N90
    :param l50_dict: dictionary of strain name: L50
    :param l75_dict: dictionary of strain name: L75
    :param l90_dict: dictionary of strain name: L90
    :param orf_dist_dict: dictionary of strain name: tuple of ORF length frequencies
    :param genus_dict: dictionary of strain name: genus
    :param sequencepath: path of folder containing FASTA genomes
    """
    # Initialise string with header information
    data = 'SampleName,TotalLength,NumContigs,LongestContig,Contigs>1000000,Contigs>500000,Contigs>100000,' \
           'Contigs>50000,Contigs>10000,Contigs>5000,Contigs<5000,TotalORFs,ORFs>3000,ORFs>1000,ORFs>500,' \
           'ORFs<500,N50,N75,N90,L50,L75,L90,GC%,Genus\n'
    # Create and open the report for writign
    with open(os.path.join(sequencepath, 'extracted_features.csv'), 'w') as feature_report:
        for file_name in sorted(longest_contig_dict):
            # Populate the data string with the appropriate values
            data += '{name},{totlen},{numcontigs},{longestcontig},{over_106},{over_56},{over_105},{over_55},' \
                    '{over_104},{over_54},{under_54},{tORFS},{ORF33},{ORF13},{ORF52}, {ORF11},{n50},{n75},{n90},' \
                    '{l50},{l75},{l90},{gc},{genus}\n'\
                .format(name=file_name,
                        totlen=genome_length_dict[file_name],
                        numcontigs=num_contigs_dict[file_name],
                        longestcontig=longest_contig_dict[file_name],
                        over_106=contig_dist_dict[file_name][0],
                        over_56=contig_dist_dict[file_name][1],
                        over_105=contig_dist_dict[file_name][2],
                        over_55=contig_dist_dict[file_name][3],
                        over_104=contig_dist_dict[file_name][4],
                        over_54=contig_dist_dict[file_name][5],
                        under_54=contig_dist_dict[file_name][6],
                        tORFS=orf_dist_dict[file_name][0],
                        ORF33=orf_dist_dict[file_name][1],
                        ORF13=orf_dist_dict[file_name][2],
                        ORF52=orf_dist_dict[file_name][3],
                        ORF11=orf_dist_dict[file_name][4],
                        n50=n50_dict[file_name],
                        n75=n75_dict[file_name],
                        n90=n90_dict[file_name],
                        l50=l50_dict[file_name],
                        l75=l75_dict[file_name],
                        l90=l90_dict[file_name],
                        gc=gc_dict[file_name],
                        genus=genus_dict[file_name])
        # Write the string to file
        feature_report.write(data)