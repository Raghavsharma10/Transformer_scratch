def get_codon(seq, codon_no, start_offset):
    """
    This function takes a sequece and a codon number and returns the codon
    found in the sequence at that position 
    """
    seq = seq.replace("-","")
    codon_start_pos = int(codon_no - 1)*3 - start_offset
    codon = seq[codon_start_pos:codon_start_pos + 3]
    return codon