def get_retained_chimeras(output_fp_de_novo_nonchimeras,
                          output_fp_ref_nonchimeras,
                          output_combined_fp,
                          chimeras_retention='union'):
    """ Gets union or intersection of two supplied fasta files

    output_fp_de_novo_nonchimeras: filepath of nonchimeras from de novo
     usearch detection.
    output_fp_ref_nonchimeras: filepath of nonchimeras from reference based
     usearch detection.
    output_combined_fp: filepath to write retained sequences to.
    chimeras_retention: accepts either 'intersection' or 'union'.  Will test
     for chimeras against the full input error clustered sequence set, and
     retain sequences flagged as non-chimeras by either (union) or
     only those flagged as non-chimeras by both (intersection)."""

    de_novo_non_chimeras = []
    reference_non_chimeras = []

    de_novo_nonchimeras_f = open(output_fp_de_novo_nonchimeras, "U")
    reference_nonchimeras_f = open(output_fp_ref_nonchimeras, "U")

    output_combined_f = open(output_combined_fp, "w")

    for label, seq in parse_fasta(de_novo_nonchimeras_f):
        de_novo_non_chimeras.append(label)
    de_novo_nonchimeras_f.close()
    for label, seq in parse_fasta(reference_nonchimeras_f):
        reference_non_chimeras.append(label)
    reference_nonchimeras_f.close()

    de_novo_non_chimeras = set(de_novo_non_chimeras)
    reference_non_chimeras = set(reference_non_chimeras)

    if chimeras_retention == 'union':
        all_non_chimeras = de_novo_non_chimeras.union(reference_non_chimeras)
    elif chimeras_retention == 'intersection':
        all_non_chimeras =\
            de_novo_non_chimeras.intersection(reference_non_chimeras)

    de_novo_nonchimeras_f = open(output_fp_de_novo_nonchimeras, "U")
    reference_nonchimeras_f = open(output_fp_ref_nonchimeras, "U")

    # Save a list of already-written labels
    labels_written = []

    for label, seq in parse_fasta(de_novo_nonchimeras_f):
        if label in all_non_chimeras:
            if label not in labels_written:
                output_combined_f.write('>%s\n%s\n' % (label, seq))
                labels_written.append(label)
    de_novo_nonchimeras_f.close()
    for label, seq in parse_fasta(reference_nonchimeras_f):
        if label in all_non_chimeras:
            if label not in labels_written:
                output_combined_f.write('>%s\n%s\n' % (label, seq))
                labels_written.append(label)
    reference_nonchimeras_f.close()

    output_combined_f.close()

    return output_combined_fp