def find_nucleotid_mismatches(sbjct_start, sbjct_seq, qry_seq, promoter = False):
    """ 
    This function takes two alligned sequence (subject and query), and the 
    position on the subject where the alignment starts. The sequences are 
    compared one nucleotide at a time. If mis matches are found they are 
    saved. If a gap is found the function find_nuc_indel is called to find
    the entire indel and it is also saved into the list mis_matches. If 
    promoter sequences are given as arguments, these are reversed the and 
    the absolut value of the sequence position  used, but when mutations
    are saved the negative value and det reverse sequences are saved in 
    mis_mathces.
    """

    # Initiate the mis_matches list that will store all found mis matcehs
    mis_matches = []
    
    sbjct_start = abs(sbjct_start)
    seq_pos = sbjct_start

    # Set variables depending on promoter status
    factor = 1
    mut_prefix = "r."
    if promoter == True:
        factor = (-1)
        mut_prefix = "n."
        # Reverse promoter sequences
        sbjct_seq = sbjct_seq[::-1]
        qry_seq = qry_seq[::-1]    
    
    # Go through sequences one nucleotide at a time
    shift = 0
    for index in range(sbjct_start - 1, len(sbjct_seq)):
        mut_name = mut_prefix
        mut = ""

        # Shift index according to gaps
        i = index + shift
        
        # If the end of the sequence is reached, stop
        if i == len(sbjct_seq):
            break
        
        sbjct_nuc = sbjct_seq[i]
        qry_nuc = qry_seq[i]
        
        # Check for mis matches
        if sbjct_nuc.upper() != qry_nuc.upper():
            
            # check for insertions and deletions
            if sbjct_nuc == "-" or qry_nuc == "-":
                if sbjct_nuc == "-":
                    mut = "ins"
                    indel_start_pos = (seq_pos -1) *factor
                    indel_end_pos = seq_pos * factor
                    indel = find_nuc_indel(sbjct_seq[i:], qry_seq[i:])
                else:
                    mut = "del"
                    indel_start_pos = seq_pos * factor
                    indel = find_nuc_indel(qry_seq[i:], sbjct_seq[i:]) 
                    indel_end_pos = (seq_pos + len(indel) - 1) * factor  
                    seq_pos += len(indel) - 1
                
                # Shift the index to the end of the indel
                shift += len(indel) - 1
                                     
                # Write mutation name, depending on sequnce
                if len(indel) == 1 and mut == "del":
                    mut_name += str(indel_start_pos) + mut + indel
                else:
                    if promoter == True:

                        # Reverse the sequence and the start and end positions
                        indel = indel[::-1]
                        temp = indel_start_pos
                        indel_start_pos = indel_end_pos
                        indel_end_pos = temp
    
                    mut_name += str(indel_start_pos) + "_" +str(indel_end_pos) + mut + indel  
                
                mis_matches += [[mut, seq_pos * factor, seq_pos * factor, indel, mut_name, mut, indel]]
            
            # Check for substitutions mutations
            else:
                mut = "sub"
                mut_name += str(seq_pos * factor) + sbjct_nuc + ">" + qry_nuc
                mis_matches += [[mut, seq_pos * factor, seq_pos * factor, qry_nuc, mut_name, sbjct_nuc, qry_nuc]]

        # Increment sequence position
        if mut != "ins":
            seq_pos += 1

    return mis_matches