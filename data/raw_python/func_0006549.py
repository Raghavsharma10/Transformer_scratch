def get_indels(sbjct_seq, qry_seq, start_pos):
    """
    This function uses regex to find inserts and deletions in sequences
    given as arguments. A list of these indels are returned. The list 
    includes, type of mutations(ins/del), subject codon no of found 
    mutation, subject sequence position, insert/deletions nucleotide 
    sequence, and the affected qry codon no.
    """

    seqs = [sbjct_seq, qry_seq]
    indels = []
    gap_obj = re.compile(r"-+")
    for i in range(len(seqs)):
        for match in gap_obj.finditer(seqs[i]):
            pos = int(match.start())
            gap = match.group()

            # Find position of the mutation corresponding to the subject sequence
            sbj_pos = len(sbjct_seq[:pos].replace("-","")) + start_pos
    
            # Get indel sequence and the affected sequences in sbjct and qry in the reading frame
            indel = seqs[abs(i-1)][pos:pos+len(gap)]                   

            # Find codon number for mutation
            codon_no = int(math.ceil((sbj_pos)/3))
            qry_pos = len(qry_seq[:pos].replace("-","")) + start_pos
            qry_codon = int(math.ceil((qry_pos)/3))
            if i == 0:
                mut = "ins"
            else:
                mut = "del"
            
            indels.append( [mut, codon_no, sbj_pos, indel, qry_codon])

    # Sort indels based on codon position and sequence position
    indels = sorted(indels, key = lambda x:(x[1],x[2]))
    
    return indels