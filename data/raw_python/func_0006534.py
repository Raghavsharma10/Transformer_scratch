def mlstreporter(self):
        """ Parse the results into a report"""
        logging.info('Writing reports')
        # Initialise variables
        header_row = str()
        combinedrow = str()
        combined_header_row = str()
        reportdirset = set()
        mlst_dict = dict()
        # Populate a set of all the report directories to use. A standard analysis will only have a single report
        # directory, while pipeline analyses will have as many report directories as there are assembled samples
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                # Ignore samples that lack a populated reportdir attribute
                if sample[self.analysistype].reportdir != 'NA':
                    make_path(sample[self.analysistype].reportdir)
                    # Add to the set - I probably could have used a counter here, but I decided against it
                    reportdirset.add(sample[self.analysistype].reportdir)
        # Create a report for each sample from :self.resultprofile
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                if sample[self.analysistype].reportdir != 'NA':
                    if type(sample[self.analysistype].allelenames) == list:
                        # Initialise the string
                        row = str()
                        if self.analysistype == 'mlst':
                            header_row = str()
                            try:
                                if sample.general.referencegenus not in mlst_dict:
                                    mlst_dict[sample.general.referencegenus] = dict()
                            except AttributeError:
                                sample.general.referencegenus = 'ND'
                                mlst_dict[sample.general.referencegenus] = dict()
                        # Additional fields such as clonal complex and lineage
                        additional_fields = list()
                        #
                        if self.meta_headers:
                            for header in self.meta_headers:
                                try:
                                    _ = sample[self.analysistype].meta_dict[
                                        sample[self.analysistype].sequencetype][header]
                                    additional_fields.append(header.rstrip())
                                except (AttributeError, KeyError):
                                    pass
                        if self.analysistype == 'mlst':
                            additional_fields = sorted(additional_fields)
                            #
                            try:
                                if sample.general.referencegenus == 'Listeria':
                                    additional_fields.append('PredictedSerogroup')
                            except AttributeError:
                                pass
                            header_fields = additional_fields
                        else:
                            additional_fields = [
                                'genus', 'species', 'subspecies', 'lineage', 'sublineage', 'other_designation', 'notes'
                            ]
                            header_fields = [
                                'rMLST_genus', 'species', 'subspecies', 'lineage', 'sublineage', 'other_designation',
                                'notes'
                            ]
                        # Populate the header with the appropriate data, including all the genes in the list of targets
                        if not header_row:
                            if additional_fields:
                                header_row = 'Strain,MASHGenus,{additional},SequenceType,Matches,{matches},\n' \
                                    .format(additional=','.join(header_fields),
                                            matches=','.join(sorted(sample[self.analysistype].allelenames)))
                            else:
                                header_row = 'Strain,MASHGenus,SequenceType,Matches,{matches},\n' \
                                    .format(matches=','.join(sorted(sample[self.analysistype].allelenames)))
                        # Iterate through the best sequence types for the sample
                        for seqtype in self.resultprofile[sample.name]:
                            sample[self.analysistype].sequencetype = seqtype
                            try:
                                if sample.general.referencegenus == 'Listeria':
                                    for serogroup, mlst_list in self.listeria_serogroup_dict.items():
                                        if seqtype in [str(string) for string in mlst_list]:
                                            sample[self.analysistype].meta_dict[seqtype]['PredictedSerogroup'] = \
                                                serogroup
                            except AttributeError:
                                pass
                            # The number of matches to the profile
                            sample[self.analysistype].matches = list(self.resultprofile[sample.name][seqtype].keys())[0]
                            # Extract the closest reference genus
                            try:
                                genus = sample.general.referencegenus
                            except AttributeError:
                                try:
                                    genus = sample.general.closestrefseqgenus
                                except AttributeError:
                                    genus = 'ND'
                            # If this is the first of one or more sequence types, include the sample name
                            if additional_fields:
                                row += '{name},{mashgenus},{additional},{seqtype},{matches},'\
                                    .format(name=sample.name,
                                            mashgenus=genus,
                                            additional=','.join(sample[self.analysistype].
                                                                meta_dict[sample[self.analysistype]
                                                                .sequencetype][header] for header in additional_fields),
                                            seqtype=seqtype,
                                            matches=sample[self.analysistype].matches)
                            else:
                                row += '{name},{mashgenus},{seqtype},{matches},' \
                                    .format(name=sample.name,
                                            mashgenus=genus,
                                            seqtype=seqtype,
                                            matches=sample[self.analysistype].matches)
                            # Iterate through all the genes present in the analyses for the sample
                            for gene in sorted(sample[self.analysistype].allelenames):
                                refallele = sample[self.analysistype].profiledata[seqtype][gene]
                                # Set the allele and percent id from the dictionary's keys and values, respectively
                                allele = \
                                    list(self.resultprofile[sample.name][seqtype][sample[self.analysistype].matches]
                                         [gene].keys())[0]
                                percentid = \
                                    list(self.resultprofile[sample.name][seqtype][sample[self.analysistype].matches]
                                         [gene].values())[0]
                                try:
                                    if refallele and refallele != allele:
                                        if 0 < float(percentid) < 100:
                                            row += '{} ({:.2f}%),'.format(allele, float(percentid))
                                        else:
                                            row += '{} ({}),'.format(allele, refallele)
                                    else:
                                        # Add the allele and % id to the row (only add the % identity if it is not 100%)
                                        if 0 < float(percentid) < 100:
                                            row += '{} ({:.2f}%),'.format(allele, float(percentid))
                                        else:
                                            row += '{},'.format(allele)
                                    self.referenceprofile[sample.name][gene] = allele
                                except ValueError:
                                    pass
                            # Add a newline
                            row += '\n'
                        #
                        combinedrow += row
                        #
                        combined_header_row += header_row
                        combined_header_row += row
                        if self.analysistype == 'mlst':
                            mlst_dict[sample.general.referencegenus]['header'] = header_row
                            try:
                                mlst_dict[sample.general.referencegenus]['combined_row'] += row
                            except KeyError:
                                mlst_dict[sample.general.referencegenus]['combined_row'] = str()
                                mlst_dict[sample.general.referencegenus]['combined_row'] += row
                        # If the length of the # of report directories is greater than 1 (script is being run as part of
                        # the assembly pipeline) make a report for each sample
                        if self.pipeline:
                            # Open the report
                            with open(os.path.join(sample[self.analysistype].reportdir,
                                                   '{}_{}.csv'.format(sample.name, self.analysistype)), 'w') as report:
                                # Write the row to the report
                                report.write(header_row)
                                report.write(row)
        # Create the report folder
        make_path(self.reportpath)
        # Create the report containing all the data from all samples
        if self.analysistype == 'mlst':
            for genus in mlst_dict:
                if mlst_dict[genus]['combined_row']:
                    with open(os.path.join(self.reportpath, '{at}_{genus}.csv'.format(at=self.analysistype,
                                                                                      genus=genus)), 'w') \
                            as mlstreport:
                        # Add the header
                        mlstreport.write(mlst_dict[genus]['header'])
                        # Write the results to this report
                        mlstreport.write(mlst_dict[genus]['combined_row'])
            with open(os.path.join(self.reportpath,  '{at}.csv'.format(at=self.analysistype)), 'w') \
                    as combinedreport:
                # Write the results to this report
                combinedreport.write(combined_header_row)
        else:
            with open(os.path.join(self.reportpath,  '{at}.csv'.format(at=self.analysistype)), 'w') \
                    as combinedreport:
                # Add the header
                combinedreport.write(header_row)
                # Write the results to this report
                combinedreport.write(combinedrow)