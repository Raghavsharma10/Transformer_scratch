def get_inframe_gap(seq, nucs_needed = 3):
    """
    This funtion takes a sequnece starting with a gap or the complementary 
    seqeuence to the gap, and the number of nucleotides that the seqeunce
    should contain in order to maintain the correct reading frame. The 
    sequence is gone through and the number of non-gap characters are 
    counted. When the number has reach the number of needed nucleotides 
    the indel is returned. If the indel is a 'clean' insert or deletion 
    that starts in the start of a codon and can be divided by 3, then only 
    the gap is returned.
    """
    nuc_count = 0
    gap_indel  = ""
    nucs = ""
    for i in range(len(seq)):

        # Check if the character is not a gap
        if seq[i] != "-":

            # Check if the indel is a 'clean' 
            # i.e. if the insert or deletion starts at the first nucleotide in the codon and can be divided by 3
            if gap_indel.count("-") == len(gap_indel) and gap_indel.count("-") >= 3 and len(gap_indel) != 0:
                return gap_indel
            nuc_count += 1
        gap_indel += seq[i]

        # If the number of nucleotides in the indel equals the amount needed for the indel, the indel is returned.
        if nuc_count == nucs_needed:
            return gap_indel

    # This will only happen if the gap is in the very end of a sequence
    return gap_indel