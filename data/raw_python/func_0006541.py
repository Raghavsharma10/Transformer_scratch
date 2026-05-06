def find_mismatches(gene, sbjct_start, sbjct_seq, qry_seq, alternative_overlaps = []):
    """
    This function finds mis matches between two sequeces. Depending on the
    the sequence type either the function find_codon_mismatches or 
    find_nucleotid_mismatches are called, if the sequences contains both 
    a promoter and a coding region both functions are called. The function 
    can also call it self if alternative overlaps is give. All found mis 
    matches are returned
    """

    # Initiate the mis_matches list that will store all found mis matcehs
    mis_matches = []

    # Find mis matches in RNA genes
    if gene in RNA_gene_list:
        mis_matches += find_nucleotid_mismatches(sbjct_start, sbjct_seq, qry_seq)
    else:
        # Check if the gene sequence is with a promoter
        regex = r"promoter_size_(\d+)(?:bp)"
        promtr_gene_objt = re.search(regex, gene)

        # Check for promoter sequences
        if promtr_gene_objt:

            # Get promoter length
            promtr_len = int(promtr_gene_objt.group(1))

            # Extract promoter sequence, while considering gaps	
            # --------agt-->----
            #    ---->?
            if sbjct_start <= promtr_len:

                #Find position in sbjct sequence where promoter ends
                promtr_end = 0
                nuc_count = sbjct_start - 1
                for i in range(len(sbjct_seq)): 
                    promtr_end += 1
                    if sbjct_seq[i] != "-":
                        nuc_count += 1
                    if nuc_count == promtr_len:
                        break    

                # Check if only a part of the promoter is found
                #--------agt-->----
                # ----
                promtr_sbjct_start = -1
                if nuc_count < promtr_len:
                    promtr_sbjct_start = nuc_count - promtr_len

                # Get promoter part of subject and query 
                sbjct_promtr_seq = sbjct_seq[:promtr_end]
                qry_promtr_seq = qry_seq[:promtr_end]

                
                # For promoter part find nucleotide mis matches
                mis_matches += find_nucleotid_mismatches(promtr_sbjct_start, sbjct_promtr_seq, qry_promtr_seq, promoter = True)
                
                # Check if gene is also found
                #--------agt-->----
                #     -----------           
                if (sbjct_start + len(sbjct_seq.replace("-", ""))) > promtr_len:
                    sbjct_gene_seq = sbjct_seq[promtr_end:]
                    qry_gene_seq = qry_seq[promtr_end:]
                    sbjct_gene_start = 1

                    # Find mismatches in gene part
                    mis_matches += find_codon_mismatches(sbjct_gene_start, sbjct_gene_seq, qry_gene_seq)
            
            # No promoter, only gene is found
            #--------agt-->----
            #            ----- 
            else:
                sbjct_gene_start = sbjct_start - promtr_len
            
                # Find mismatches in gene part
                mis_matches += find_codon_mismatches(sbjct_gene_start, sbjct_seq, qry_seq)
            
        else:
            # Find mismatches in gene
            mis_matches += find_codon_mismatches(sbjct_start, sbjct_seq, qry_seq)

    # Find mismatches in alternative overlaps if any
    for overlap in alternative_overlaps:
        mis_matches += find_mismatches(gene, overlap[0], overlap[2], overlap[3])

    return mis_matches