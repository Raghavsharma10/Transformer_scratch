def reporter(self, analysistype='genesippr'):
        """
        Creates a report of the genesippr results
        :param analysistype: The variable to use when accessing attributes in the metadata object
        """
        logging.info('Creating {} report'.format(analysistype))
        # Create a dictionary to link all the genera with their genes
        genusgenes = dict()
        # The organism-specific targets are in .tfa files in the target path
        targetpath = str()
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                targetpath = sample[analysistype].targetpath
        for organismfile in glob(os.path.join(targetpath, '*.tfa')):
            organism = os.path.splitext(os.path.basename(organismfile))[0]
            # Use BioPython to extract all the gene names from the file
            for record in SeqIO.parse(open(organismfile), 'fasta'):
                # Append the gene names to the genus-specific list
                try:
                    genusgenes[organism].add(record.id.split('_')[0])
                except (KeyError, IndexError):
                    genusgenes[organism] = set()
                    genusgenes[organism].add(record.id.split('_')[0])
        # Determine from which genera the gene hits were sourced
        for sample in self.runmetadata.samples:
            # Initialise the list to store the genera
            sample[analysistype].targetgenera = list()
            if sample.general.bestassemblyfile != 'NA':
                for organism in genusgenes:
                    # Iterate through all the genesippr hits and attribute each gene to the appropriate genus
                    for gene in sample[analysistype].results:
                        # If the gene name is in the genes from that organism, add the genus name to the list of
                        # genera found in the sample
                        if gene.split('_')[0] in genusgenes[organism]:
                            if organism not in sample[analysistype].targetgenera:
                                sample[analysistype].targetgenera.append(organism)
        # Create the path in which the reports are stored
        make_path(self.reportpath)
        # The report will have every gene for all genera in the header
        header = 'Strain,Genus,{},\n'.format(','.join(self.genelist))
        data = str()
        with open(os.path.join(self.reportpath, analysistype + '.csv'), 'w') as report:
            for sample in self.runmetadata.samples:
                sample[analysistype].report_output = list()
                if sample.general.bestassemblyfile != 'NA':
                    # Add the genus/genera found in the sample
                    data += '{},{},'.format(sample.name, ';'.join(sample[analysistype].targetgenera))
                    best_dict = dict()
                    if sample[analysistype].results:
                        gene_check = list()
                        # Find the best match for all the hits
                        for target, pid in sample[analysistype].results.items():
                            gene_name = target.split('_')[0]
                            for gene in self.genelist:
                                # If the key matches a gene in the list of genes
                                if gene == gene_name:
                                    # If the percent identity is better, update the dictionary
                                    try:
                                        if float(pid) > best_dict[gene]:
                                            best_dict[gene] = float(pid)
                                    except KeyError:
                                        best_dict[gene] = float(pid)
                        for gene in self.genelist:
                            # If the gene was not found in the sample, print an empty cell in the report
                            try:
                                best_dict[gene]
                            except KeyError:
                                data += ','
                            # Print the required information for the gene
                            for name, identity in sample[analysistype].results.items():
                                if name.split('_')[0] == gene and gene not in gene_check:
                                    data += '{pid}%'.format(pid=best_dict[gene])
                                    try:
                                        if not sample.general.trimmedcorrectedfastqfiles[0].endswith('.fasta'):
                                            data += ' ({avgd} +/- {std}),'\
                                                .format(avgd=sample[analysistype].avgdepth[name],
                                                        std=sample[analysistype].standarddev[name])
                                        else:
                                            data += ','
                                    except IndexError:
                                        data += ','
                                    gene_check.append(gene)
                                    # Add the simplified results to the object - used in the assembly pipeline report
                                    sample[analysistype].report_output.append(gene)
                        # Add a newline after each sample
                        data += '\n'
                    # Add a newline if the sample did not have any gene hits
                    else:
                        data += '\n'
            # Write the header and data to file
            report.write(header)
            report.write(data)