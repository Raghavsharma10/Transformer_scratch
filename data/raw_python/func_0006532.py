def profiler(self):
        """Creates a dictionary from the profile scheme(s)"""
        logging.info('Loading profiles')
        # Initialise variables
        profiledata = defaultdict(make_dict)
        reverse_profiledata = dict()
        profileset = set()
        # Find all the unique profiles to use with a set
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                if sample[self.analysistype].profile != 'NA':
                    profileset.add(sample[self.analysistype].profile)
        # Extract the profiles for each set
        for sequenceprofile in profileset:
            #
            if sequenceprofile not in self.meta_dict:
                self.meta_dict[sequenceprofile] = dict()
            reverse_profiledata[sequenceprofile] = dict()
            self.meta_dict[sequenceprofile]['ND'] = dict()
            # Clear the list of genes
            geneset = set()
            # Calculate the total number of genes in the typing scheme
            for sample in self.runmetadata.samples:
                if sample.general.bestassemblyfile != 'NA':
                    if sequenceprofile == sample[self.analysistype].profile:
                        geneset = {allele for allele in sample[self.analysistype].alleles}
            try:
                # Open the sequence profile file as a dictionary
                profile = DictReader(open(sequenceprofile), dialect='excel-tab')
            # Revert to standard comma separated values
            except KeyError:
                # Open the sequence profile file as a dictionary
                profile = DictReader(open(sequenceprofile))
            # Iterate through the rows
            for row in profile:
                # Populate the profile dictionary with profile number: {gene: allele}. Use the first field name,
                # which will be either ST, or rST as the key to determine the profile number value
                allele_comprehension = {gene: allele for gene, allele in row.items() if gene in geneset}
                st = row[profile.fieldnames[0]]
                for header, value in row.items():
                    value = value if value else 'ND'
                    if header not in geneset and header not in ['ST', 'rST']:
                        if st not in self.meta_dict[sequenceprofile]:
                            self.meta_dict[sequenceprofile][st] = dict()
                        if header == 'CC' or header == 'clonal_complex':
                            header = 'CC'
                        self.meta_dict[sequenceprofile][st][header] = value
                        self.meta_dict[sequenceprofile]['ND'][header] = 'ND'
                        self.meta_dict[sequenceprofile][st]['PredictedSerogroup'] = 'ND'
                        if header not in self.meta_headers:
                            self.meta_headers.append(header)
                profiledata[sequenceprofile][st] = allele_comprehension
                # Create a 'reverse' dictionary using the the allele comprehension as the key, and
                # the sequence type as the value - can be used if exact matches are ever desired
                reverse_profiledata[sequenceprofile].update({frozenset(allele_comprehension.items()): st})
            # Add the profile data, and gene list to each sample
            for sample in self.runmetadata.samples:
                if sample.general.bestassemblyfile != 'NA':
                    if sequenceprofile == sample[self.analysistype].profile:
                        # Populate the metadata with the profile data
                        sample[self.analysistype].profiledata = profiledata[sample[self.analysistype].profile]
                        sample[self.analysistype].reverse_profiledata = reverse_profiledata[sequenceprofile]
                        sample[self.analysistype].meta_dict = self.meta_dict[sequenceprofile]
                else:
                    sample[self.analysistype].profiledata = 'NA'
                    sample[self.analysistype].reverse_profiledata = 'NA'
                    sample[self.analysistype].meta_dict = 'NA'