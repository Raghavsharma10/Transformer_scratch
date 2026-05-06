def write_output(gene, gene_name, mis_matches, known_mutations, known_stop_codon, unknown_flag, GENES):
    """
    This function takes a gene name a list of mis matches found betreewn subject and query of
    this gene, the dictionary of known mutation in the point finder database, and the flag telling 
    weather the user wants unknown mutations to be reported.
    All mis matches are looked up in the known mutation dict to se if the mutation is known, 
    and in this case what drug resistence it causes.
    The funtions returns a 3 strings that are used as output to the users.
    One string is only tab seperated and contains the mutations listed line by line. 
    If the unknown flag is set to true it will contain both known and unknown mutations. 
    The next string contains only known mutation and are given in in a format that is easy to
    convert to HTML. The last string is the HTML tab sting from the unknown mutations.
    """
    RNA = False
    known_header = "Mutation\tNucleotide change\tAmino acid change\tResistance\tPMID\n"
    unknown_header = "Mutation\tNucleotide change\tAmino acid change\n"
    if gene in RNA_gene_list:
        RNA = True
        known_header = "Mutation\tNucleotide change\tResistance\tPMID\n"
        unknown_header = "Mutation\tNucleotide change\n"

    known_lst = []
    unknown_lst = []
    all_results_lst = [] 
    output_mut = []
    stop_codons = []

    # Go through each mutation    
    for i in range(len(mis_matches)):
        m_type = mis_matches[i][0]
        pos = mis_matches[i][1] # sort on pos?
        look_up_pos = mis_matches[i][2]
        look_up_mut = mis_matches[i][3]
        mut_name = mis_matches[i][4]
        nuc_ref = mis_matches[i][5]
        nuc_alt = mis_matches[i][6]
        ref =  mis_matches[i][-2]
        alt = mis_matches[i][-1]

        # First index in list indicates if mutation is known
        output_mut += [[]]
        #output_mut[i] = [0]

    	# Define output vaiables
        codon_change = nuc_ref + " -> " + nuc_alt
        aa_change = ref + " -> " + alt
        if RNA == True:
            aa_change = "RNA mutations"
        elif pos < 0:
            aa_change = "Promoter mutations"
        
        # Check if mutation is known
        gene_mut_name, resistence, pmid = look_up_known_muts(known_mutations, known_stop_codon, gene, look_up_pos, look_up_mut, m_type, gene_name, mut_name)
        gene_mut_name = gene_mut_name + " " + mut_name

        output_mut[i] = [gene_mut_name, codon_change, aa_change, resistence, pmid]
        
        # Add mutation to output strings for known mutations 
        if resistence != "Unknown":
            if RNA == True:
                # don't include the amino acid change field for RNA mutations
                known_lst += ["\t".join(output_mut[i][:2]) + "\t" + "\t".join(output_mut[i][3:])]
            else:
                known_lst += ["\t".join(output_mut[i])]
            all_results_lst += ["\t".join(output_mut[i])]

        # Add mutation to output strings for unknown mutations 
        else:
            if RNA == True:
                unknown_lst += ["\t".join(output_mut[i][:2])]
            else:
                unknown_lst += ["\t".join(output_mut[i][:3])]
            if unknown_flag == True:
                all_results_lst += ["\t".join(output_mut[i])]

        # Check that you do not print two equal lines (can happen it two indels occure in the same codon)
        if len(output_mut) > 1:
            if output_mut[i] == output_mut[i-1]:
                if resistence != "Unknown":
                    known_lst = known_lst[:-1]
                    all_results_lst = all_results_lst[:-1]
                else:
                    unknown_lst = unknown_lst[:-1]
                    if unknown_flag == True:
                        all_results_lst = all_results_lst[:-1]
        if "Premature stop codon" in mut_name:
            sbjct_len = GENES[gene]['sbjct_len']
            qry_len = pos * 3 
            prec_truckat = round(((float(sbjct_len) - qry_len )/ float(sbjct_len)) * 100, 2) 
            perc = "%"
            stop_codons.append("Premature stop codon in %s, %.2f%s lost"%(gene, prec_truckat, perc))

    # Creat final strings
    all_results = "\n".join(all_results_lst)
    total_known_str = "" 
    total_unknown_str = ""

    # Check if there are only unknown mutations
    resistence_lst = [res for mut in output_mut for res in mut[3].split(",")]

    # Save known mutations
    unknown_no = resistence_lst.count("Unknown")
    if unknown_no < len(resistence_lst):
        total_known_str = known_header + "\n".join(known_lst)
    else:
        total_known_str = "No known mutations found in %s"%gene_name

    # Save unknown mutations
    if unknown_no > 0:
        total_unknown_str = unknown_header + "\n".join(unknown_lst)
    else:
        total_unknown_str = "No unknown mutations found in %s"%gene_name

    return all_results, total_known_str, total_unknown_str, resistence_lst + stop_codons