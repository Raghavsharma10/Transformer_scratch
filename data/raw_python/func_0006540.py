def find_best_sequence(hits_found, specie_path, gene, silent_N_flag):
    """
    This function takes the list hits_found as argument. This contains all 
    hits found for the blast search of one gene. A hit includes the subjct 
    sequence, the query, and the start and stop position of the allignment 
    corresponding to the subject sequence. This function finds the best 
    hit by concatinating sequences of found hits. If different overlap 
    sequences occurr these are saved in the list alternative_overlaps. The 
    subject and query sequence of the concatinated sequence to gether with 
    alternative overlaps and the corresponding start stop
    positions are returned.
    """

    # Get information from the fisrt hit found	
    all_start = hits_found[0][0]
    current_end = hits_found[0][1]
    final_sbjct = hits_found[0][2]
    final_qry = hits_found[0][3] 
    sbjct_len = hits_found[0][4]  

    alternative_overlaps = []
    
    # Check if more then one hit was found within the same gene
    for i in range(len(hits_found)-1):

        # Save information from previous hit
        pre_block_start = hits_found[i][0]
        pre_block_end = hits_found[i][1]
        pre_sbjct = hits_found[i][2]
        pre_qry = hits_found[i][3]

	# Save information from next hit
        next_block_start = hits_found[i+1][0]
        next_block_end = hits_found[i+1][1]
        next_sbjct = hits_found[i+1][2]
        next_qry = hits_found[i+1][3]

        # Check for overlapping sequences, collaps them and save alternative overlaps if any
        if next_block_start <= current_end:

            # Find overlap start and take gaps into account     
            pos_count = 0
            overlap_pos = pre_block_start
            for i in range(len(pre_sbjct)):

                # Stop loop if overlap_start position is reached
                if overlap_pos == next_block_start:
                    overlap_start = pos_count
                    break
                if pre_sbjct[i] != "-":
                    overlap_pos += 1
                pos_count += 1
            
            # Find overlap length and add next sequence to final sequence 
            if len(pre_sbjct[overlap_start:]) > len(next_sbjct):
                #  <--------->
                #     <--->
                overlap_len = len(next_sbjct)
                overlap_end_pos = next_block_end
            else:
                #  <--------->
                #        <--------->
                overlap_len = len(pre_sbjct[overlap_start:])
                overlap_end_pos = pre_block_end

                # Update current end
                current_end = next_block_end

                # Use the entire pre sequence and add the last part of the next sequence
                final_sbjct += next_sbjct[overlap_len:]
                final_qry += next_qry[overlap_len:]
                
            # Find query overlap sequences
            pre_qry_overlap = pre_qry[overlap_start : (overlap_start + overlap_len)] # can work for both types of overlap
            next_qry_overlap = next_qry[:overlap_len]
            sbjct_overlap = next_sbjct[:overlap_len]

            # If alternative query overlap excist save it
            if pre_qry_overlap != next_qry_overlap:
                print("OVERLAP WARNING:")
                print(pre_qry_overlap, "\n", next_qry_overlap)

                # Save alternative overlaps
                alternative_overlaps += [(next_block_start, overlap_end_pos, sbjct_overlap, next_qry_overlap)]
        
        elif next_block_start > current_end:
            #  <------->
            #              <-------> 
            gap_size = next_block_start - current_end - 1
            final_qry += "N"*gap_size
            if silent_N_flag:
                final_sbjct += "N"*gap_size
            else:
                ref_seq = get_gene_seqs(specie_path, gene)
                final_sbjct += ref_seq[pre_block_end:pre_block_end+gap_size]

            current_end = next_block_end
            final_sbjct += next_sbjct
            final_qry += next_qry
    
    # Calculate coverage
    no_call = final_qry.upper().count("N")
    coverage = (current_end - all_start +1 - no_call) / float(sbjct_len)
    
    # Calculate identity
    equal = 0
    not_equal = 0
    for i in range(len(final_qry)):
        if final_qry[i].upper() != "N":
            if final_qry[i].upper() == final_sbjct[i].upper():
                equal += 1
            else:
                not_equal += 1
    identity = equal/float(equal + not_equal)

    return final_sbjct, final_qry, all_start, current_end, alternative_overlaps, coverage, identity