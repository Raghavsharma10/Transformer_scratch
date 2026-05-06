def get_fasta_from_uc_file(fasta_filepath,
                           uc_filepath,
                           hit_type="H",
                           output_fna_filepath=None,
                           label_prefix="",
                           output_dir=None):
    """ writes fasta of sequences from uc file of type hit_type

    fasta_filepath:  Filepath of original query fasta file
    uc_filepath:  Filepath of .uc file created by usearch post error filtering
    hit_type: type to read from first field of .uc file, "H" for hits, "N" for
     no hits.
    output_fna_filepath = fasta output filepath
    label_prefix = Added before each fasta label, important when doing ref
     based OTU picking plus de novo clustering to preserve label matching.
    output_dir: output directory
    """

    hit_type_index = 0
    seq_label_index = 8
    target_label_index = 9

    labels_hits = {}
    labels_to_keep = []

    for line in open(uc_filepath, "U"):
        if line.startswith("#") or len(line.strip()) == 0:
            continue
        curr_line = line.split('\t')
        if curr_line[0] == hit_type:
            labels_hits[curr_line[seq_label_index]] =\
                curr_line[target_label_index].strip()
            labels_to_keep.append(curr_line[seq_label_index])

    labels_to_keep = set(labels_to_keep)

    out_fna = open(output_fna_filepath, "w")

    for label, seq in parse_fasta(open(fasta_filepath, "U")):
        if label in labels_to_keep:
            if hit_type == "H":
                out_fna.write(">" + labels_hits[label] + "\n%s\n" % seq)
            if hit_type == "N":
                out_fna.write(">" + label + "\n%s\n" % seq)

    return output_fna_filepath, labels_hits