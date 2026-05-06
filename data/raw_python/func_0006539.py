def KMA(inputfile_1, gene_list, kma_db, out_path, sample_name, min_cov, mapping_path):
    """
    This function is called when KMA is the method of choice. The 
    function calls kma externally and waits for it to finish. 
    The kma output files with the prefixes .res and .aln are parsed 
    throught to obtain the required alignment informations. The subject 
    and query sequences as well as the start and stop position, 
    coverage, and subject length are stored in a results directory
    which is returned in the end.
    """
    
    # Get full path to input of output files
    inputfile_1 = os.path.abspath(inputfile_1)

    kma_outfile = os.path.abspath(out_path + "/kma_out_" + sample_name)

    kma_cmd = "%s -i %s -t_db %s -o %s -1t1 -gapopen -5 -gapextend -2 -penalty -3 -reward 1"%(mapping_path, inputfile_1, kma_db, kma_outfile) # -ID 90
    
    # Call KMA
    os.system(kma_cmd)
    if os.path.isfile(kma_outfile + ".aln") == False:
        os.system(kma_cmd)

    # Fetch kma output files
    align_filename = kma_outfile + ".aln"
    res_filename = kma_outfile + ".res"
    
    results = dict()

    # Open KMA result file
    with open(res_filename, "r") as res_file:
        header = res_file.readline()

        # Parse through each line
        for line in res_file:
            data = [data.strip() for data in line.split("\t")]
            gene = data[0]

            # Check if gene one of the user specified genes  
            if gene not in gene_list:
                continue

            # Store subject length and coverage
            sbjct_len = int(data[3])
            identity = float(data[6])
            coverage = float(data[7])

            # Result dictionary assumes that more hits can occur
            if gene not in results:
                hit = '1'
                results[gene] = dict()

            # Gene will only be there once with KMA
            else:
                hit = str(len(results[gene])) +1

            results[gene][hit] = dict()
            results[gene][hit]['sbjct_length'] = sbjct_len
            results[gene][hit]['coverage'] = coverage / 100
            results[gene][hit]["sbjct_string"] = []
            results[gene][hit]["query_string"] = []
            results[gene][hit]["homology"] = []
            results[gene][hit]['identity'] = identity

    # Open KMA alignment file   
    with open(align_filename, "r") as align_file:
        hit_no = dict()
        gene = ""

        # Parse through alignments
        for line in align_file:

            # Check when a new gene alignment start
            if line.startswith("#"):
                gene = line[1:].strip()
                if gene not in hit_no:
                    hit_no[gene] = str(1)
                else:
                    hit_no[gene] += str(int(hit_no[gene]) + 1)

            else:
                # Check if gene is one of the user specified genes             
                if gene in results:
                    if hit_no[gene] not in results[gene]:
                        sys.exit("Unexpected database redundency")
                    line_data = line.split("\t")[-1].strip()
                    if line.startswith("template"):
                        results[gene][hit_no[gene]]["sbjct_string"] += [line_data]
                    elif line.startswith("query"):
                        results[gene][hit_no[gene]]["query_string"] += [line_data]
                    else:
                        results[gene][hit_no[gene]]["homology"] += [line_data]
    
    # Concatinate all sequences lists and find subject start and subject end
    seq_start_search_str = re.compile("^-*(\w+)")
    seq_end_search_str = re.compile("\w+(-*)$")
    for gene in gene_list:
        if gene in results:
            for hit in results[gene]:
                results[gene][hit]['sbjct_string'] = "".join(results[gene][hit]['sbjct_string'])
                results[gene][hit]['query_string'] = "".join(results[gene][hit]['query_string'])
                results[gene][hit]['homology'] = "".join(results[gene][hit]['homology'])
 
   	        
                seq_start_object = seq_start_search_str.search(results[gene][hit]['query_string'])
                sbjct_start = seq_start_object.start(1) + 1

                seq_end_object = seq_end_search_str.search(results[gene][hit]['query_string'])
                sbjct_end = seq_end_object.start(1) + 1

                results[gene][hit]['query_string'] = results[gene][hit]['query_string'][sbjct_start-1:sbjct_end-1]
                results[gene][hit]['sbjct_string'] = results[gene][hit]['sbjct_string'][sbjct_start-1:sbjct_end-1]


                #if sbjct_start:
                results[gene][hit]["sbjct_start"] = sbjct_start
                results[gene][hit]["sbjct_end"] = sbjct_end
        else:
           results[gene] = ""

    return results