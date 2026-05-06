def find_nuc_indel(gapped_seq, indel_seq):
    """
    This function finds the entire indel missing in from a gapped sequence 
    compared to the indel_seqeunce. It is assumes that the sequences start
    with the first position of the gap.
    """
    ref_indel = indel_seq[0]
    for j in range(1,len(gapped_seq)):
        if gapped_seq[j] == "-":
            ref_indel += indel_seq[j]
        else:
            break
    return ref_indel