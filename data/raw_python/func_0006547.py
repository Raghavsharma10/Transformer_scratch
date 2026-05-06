def name_indel_mutation(sbjct_seq, indel, sbjct_rf_indel, qry_rf_indel, codon_no, mut, start_offset):
    """
    This function serves to name the individual mutations dependently on 
    the type of the mutation.
    """
    # Get the subject and query sequences without gaps
    sbjct_nucs = sbjct_rf_indel.replace("-", "")
    qry_nucs = qry_rf_indel.replace("-", "")

    # Translate nucleotides to amino acids
    aa_ref = ""
    aa_alt = ""
    for i in range(0, len(sbjct_nucs), 3):
        aa_ref += aa(sbjct_nucs[i:i+3])
    for i in range(0, len(qry_nucs), 3):
        aa_alt += aa(qry_nucs[i:i+3])

    # Identify the gapped sequence 
    if mut == "ins":
        gapped_seq = sbjct_rf_indel
    else:
        gapped_seq = qry_rf_indel
    gap_size = gapped_seq.count("-")

    # Write mutation names
    if gap_size < 3 and len(sbjct_nucs) ==3 and len(qry_nucs) == 3:

        # Write mutation name for substitution mutation
        mut_name = "p.%s%d%s"%(aa(sbjct_nucs), codon_no, aa(qry_nucs))

    elif len(gapped_seq) == gap_size:
        if mut == "ins":

            # Write mutation name for insertion mutation
            mut_name = name_insertion(sbjct_seq, codon_no, sbjct_nucs, aa_alt, start_offset)
            aa_ref = mut
        else:

            # Write mutation name for deletion mutation
            mut_name = name_deletion(sbjct_seq, sbjct_rf_indel, sbjct_nucs, codon_no, aa_alt, start_offset, mutation = "del")
            aa_alt = mut

    # Check for delins - mix of insertion and deletion
    else:

        # Write mutation name for a mixed insertion and deletion mutation
        mut_name = name_deletion(sbjct_seq, sbjct_rf_indel, sbjct_nucs, codon_no, aa_alt, start_offset, mutation = "delins")

    # Check for frameshift
    if gapped_seq.count("-")%3 != 0:
        # Add the frameshift tag to mutation name
        mut_name += " - Frameshift"

    return mut_name, aa_ref, aa_alt