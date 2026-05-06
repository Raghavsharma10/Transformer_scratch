def get_db_mutations(mut_db_path, gene_list, res_stop_codons):
    """
    This function opens the file resistenss-overview.txt, and reads the
    content into a dict of dicts. The dict will contain information about
    all known mutations given in the database. This dict is returned.
    """

    # Open resistens-overview.txt
    try:
        drugfile = open(mut_db_path, "r")
    except:
        sys.exit("Wrong path: %s"%(mut_db_path))
    
    # Initiate variables
    known_mutations = dict()
    drug_genes = dict()
    known_stop_codon = dict()
    indelflag = False
    stopcodonflag = False
   
    # Go throug mutation file line by line
    for line in drugfile:
        # Ignore headers and check where the indel section starts
        if line.startswith("#"):
            if "indel" in line.lower():
                indelflag = True
            elif "stop codon" in line.lower():
                stopcodonflag = True
            else:
                stopcodonflag = False
            continue
        # Ignore empty lines 
        if line.strip() == "":
            continue
        # Assert that all lines have the correct set of columns
        mutation = [data.strip() for data in line.strip().split("\t")]
        assert len(mutation) == 9, "mutation overview file (%s) must have 9 columns, %s"%(mut_db_path, mutation)

        # Extract all info on the line (even though it is not all used)
        gene_ID = mutation[0]

        # Only consider mutations in genes found in the gene list
        if gene_ID in gene_list:
            gene_name = mutation[1]
            no_of_mut = int(mutation[2])
            mut_pos = int(mutation[3])
            ref_codon = mutation[4]
            ref_aa = mutation[5]
            alt_aa = mutation[6].split(",")
            res_drug = mutation[7].replace("\t", " ")
            pmid = mutation[8].split(",")

            # Check if resistance is known to be caused by a stop codon in the gene
            if ("*" in alt_aa and res_stop_codons != 'specified') or (res_stop_codons == 'specified' and stopcodonflag == True):
                if gene_ID not in known_stop_codon:
                    known_stop_codon[gene_ID] = {"pos": [], "drug": res_drug}
                known_stop_codon[gene_ID]["pos"].append(mut_pos)

            # Add genes associated with drug resistance to drug_genes dict
            drug_lst = res_drug.split(",")
            for drug in drug_lst:
                drug = drug.upper()
                if drug not in drug_genes:
                    drug_genes[drug] = []
                if gene_ID not in drug_genes[drug]:
                    drug_genes[drug].append(gene_ID)

            # Initiate empty dict to store relevant mutation information
            mut_info = dict()
            
            # Save need mutation info with pmid cooresponding to the amino acid change
            for i in range(len(alt_aa)):
                try:
                    mut_info[alt_aa[i]] = {"gene_name": gene_name, "drug": res_drug, "pmid": pmid[i]}
                except IndexError:
                    mut_info[alt_aa[i]] = {"gene_name": gene_name, "drug": res_drug, "pmid": "-"}
    
    	    # Check if more than one mutations is needed for resistance
            if no_of_mut != 1:
                print("More than one mutation is needed, this is not implemented",  mutation)
    
            # Add all possible types of mutations to the dict
            if gene_ID not in known_mutations:
                known_mutations[gene_ID] = {"sub" : dict(), "ins" : dict(), "del" : dict()}

            # Check for the type of mutation
            if indelflag == False:
                mutation_type = "sub"
            else:
                mutation_type = ref_aa

    	    # Save mutations positions with required information given in mut_info
            if mut_pos not in known_mutations[gene_ID][mutation_type]:
                known_mutations[gene_ID][mutation_type][mut_pos] = dict() 
            for aa in alt_aa:
                known_mutations[gene_ID][mutation_type][mut_pos][aa] = mut_info[aa]

    drugfile.close()

    # Check that all genes in the gene list has known mutations
    for gene in gene_list:
        if gene not in known_mutations:
            known_mutations[gene] = {"sub" : dict(), "ins" : dict(), "del" : dict()}
    return known_mutations, drug_genes, known_stop_codon