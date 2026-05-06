def find_codon_mismatches(sbjct_start, sbjct_seq, qry_seq):
    """
    This function takes two alligned sequence (subject and query), and
    the position on the subject where the alignment starts. The sequences 
    are compared codon by codon. If a mis matches is found it is saved in 
    'mis_matches'. If a gap is found the function get_inframe_gap is used 
    to find the indel sequence and keep the sequence in the correct 
    reading frame. The function translate_indel is used to name indel 
    mutations and translate the indels to amino acids
    The function returns a list of tuples containing all needed informations
    about the mutation in order to look it up in the database dict known 
    mutation and the with the output files the the user.  
    """
    mis_matches = []
    
    # Find start pos of first codon in frame, i_start
    codon_offset = (sbjct_start-1) % 3
    i_start = 0
    if codon_offset != 0:
        i_start = 3 - codon_offset
    sbjct_start = sbjct_start + i_start
    
    # Set sequences in frame
    sbjct_seq = sbjct_seq[i_start:]
    qry_seq = qry_seq[i_start:]
    
    # Find codon number of the first codon in the sequence, start at 0
    codon_no = int((sbjct_start-1) / 3) # 1,2,3 start on 0
  
    # s_shift and q_shift are used when gaps appears
    q_shift = 0
    s_shift = 0
    mut_no = 0
    
    # Find inserts and deletions in sequence
    indel_no = 0
    indels = get_indels(sbjct_seq, qry_seq, sbjct_start)

    # Go through sequence and save mutations when found
    for index in range(0, len(sbjct_seq), 3):
        # Count codon number
        codon_no += 1
        
        # Shift index according to gaps
        s_i = index + s_shift
        q_i = index + q_shift

        # Get codons
        sbjct_codon = sbjct_seq[s_i:s_i+3]
        qry_codon =  qry_seq[q_i:q_i+3]
        
        if len(sbjct_seq[s_i:].replace("-","")) + len(qry_codon[q_i:].replace("-","")) < 6:
            break

        # Check for mutations
        if sbjct_codon.upper() != qry_codon.upper():

            # Check for codon insertions and deletions and frameshift mutations
            if "-" in sbjct_codon or "-" in qry_codon:

                # Get indel info
                try:
                    indel_data = indels[indel_no]
                except IndexError:
                    print(sbjct_codon, qry_codon)
                    print(indels)
                    print(gene, indel_data, indel_no)
                mut = indel_data[0]
                codon_no_indel = indel_data[1]                
                seq_pos = indel_data[2] + sbjct_start - 1
                indel = indel_data[3]
                indel_no +=1
                
                # Get the affected sequence in frame for both for sbjct and qry 
                if mut == "ins":
                    sbjct_rf_indel = get_inframe_gap(sbjct_seq[s_i:], 3)
                    qry_rf_indel = get_inframe_gap(qry_seq[q_i:], int(math.floor(len(sbjct_rf_indel)/3) *3))                    
                else:
                    qry_rf_indel = get_inframe_gap(qry_seq[q_i:], 3)
                    sbjct_rf_indel = get_inframe_gap(sbjct_seq[s_i:], int(math.floor(len(qry_rf_indel)/3) *3))
                                        
                mut_name, aa_ref, aa_alt = name_indel_mutation(sbjct_seq, indel, sbjct_rf_indel, qry_rf_indel, codon_no, mut, sbjct_start - 1)

                # Set index to the correct reading frame after the indel gap 
                shift_diff_before = abs(s_shift - q_shift)
                s_shift += len(sbjct_rf_indel) - 3
                q_shift += len(qry_rf_indel) - 3
                shift_diff = abs(s_shift - q_shift)

                if shift_diff_before != 0 and shift_diff %3 == 0:

                    if s_shift > q_shift:
                        nucs_needed = int((len(sbjct_rf_indel)/3) *3) + shift_diff
                        pre_qry_indel = qry_rf_indel
                        qry_rf_indel = get_inframe_gap(qry_seq[q_i:], nucs_needed)
                        q_shift += len(qry_rf_indel) - len(pre_qry_indel)
                    elif q_shift > s_shift:
                        nucs_needed = int((len(qry_rf_indel)/3)*3) + shift_diff
                        pre_sbjct_indel = sbjct_rf_indel
                        sbjct_rf_indel = get_inframe_gap(sbjct_seq[s_i:], nucs_needed)
                        s_shift += len(sbjct_rf_indel) - len(pre_sbjct_indel)

                    
                    mut_name, aa_ref, aa_alt = name_indel_mutation(sbjct_seq, indel, sbjct_rf_indel, qry_rf_indel, codon_no, mut, sbjct_start - 1) 

                    if "Frameshift" in mut_name:
                        mut_name = mut_name.split("-")[0] + "- Frame restored"

                mis_matches += [[mut, codon_no_indel, seq_pos, indel, mut_name, sbjct_rf_indel, qry_rf_indel, aa_ref, aa_alt]]

                # Check if the next mutation in the indels list is in the current codon
                # Find the number of individul gaps in the evaluated sequence
                no_of_indels = len(re.findall("\-\w", sbjct_rf_indel)) + len(re.findall("\-\w", qry_rf_indel))
                if no_of_indels > 1:

                    for j in range(indel_no, indel_no + no_of_indels - 1):
                        try:
                            indel_data = indels[j]
                        except IndexError:
                            sys.exit("indel_data list is out of range, bug!")
                        mut = indel_data[0]
                        codon_no_indel = indel_data[1]                
                        seq_pos = indel_data[2] + sbjct_start - 1
                        indel = indel_data[3]
                        indel_no +=1
                        mis_matches += [[mut, codon_no_indel, seq_pos, indel, mut_name, sbjct_rf_indel, qry_rf_indel, aa_ref, aa_alt]]

                # Set codon number, and save nucleotides from out of frame mutations                
                if mut == "del":
                    codon_no += int((len(sbjct_rf_indel) - 3)/3)
                # If evaluated insert is only gaps codon_no should not increment
                elif sbjct_rf_indel.count("-") == len(sbjct_rf_indel):
                    codon_no -= 1
            
            # Check of point mutations
            else:
                mut = "sub"
                aa_ref = aa(sbjct_codon)
                aa_alt = aa(qry_codon)
                
                if aa_ref != aa_alt:
                    # End search for mutation if a premature stop codon is found
                    mut_name = "p." + aa_ref + str(codon_no) + aa_alt

                    mis_matches += [[mut, codon_no, codon_no, aa_alt, mut_name, sbjct_codon, qry_codon, aa_ref, aa_alt]]

            # If a Premature stop codon occur report it an stop the loop
            try:
                if mis_matches[-1][-1] == "*":
                    mut_name += " - Premature stop codon"
                    mis_matches[-1][4] = mis_matches[-1][4].split("-")[0] + " - Premature stop codon"
                    break
            except IndexError:
                pass

    # Sort mutations on position
    mis_matches = sorted(mis_matches, key = lambda x:x[1])
    
    return mis_matches