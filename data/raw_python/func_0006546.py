def name_insertion(sbjct_seq, codon_no, sbjct_nucs, aa_alt, start_offset):
    """
    This function is used to name a insertion mutation based on the HGVS 
    recommendation. 
    """
    start_codon_no = codon_no - 1
    if len(sbjct_nucs) == 3:
        start_codon_no = codon_no
    start_codon = get_codon(sbjct_seq, start_codon_no, start_offset)
    end_codon = get_codon(sbjct_seq, codon_no, start_offset)
    pos_name = "p.%s%d_%s%dins%s"%(aa(start_codon), start_codon_no, aa(end_codon), codon_no, aa_alt)
    return pos_name