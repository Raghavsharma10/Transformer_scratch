def gdcsreporter(self, analysistype='GDCS'):
        """
        Creates a report of the GDCS results
        :param analysistype: The variable to use when accessing attributes in the metadata object
        """
        logging.info('Creating {} report'.format(analysistype))
        # Initialise list to store all the GDCS genes, and genera in the analysis
        gdcs = list()
        genera = list()
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                if os.path.isdir(sample[analysistype].targetpath):
                    # Update the fai dict with all the genes in the analysis, rather than just those with baited hits
                    self.gdcs_fai(sample)
                    sample[analysistype].createreport = True
                    # Determine which genera are present in the analysis
                    if sample.general.closestrefseqgenus not in genera:
                        genera.append(sample.general.closestrefseqgenus)
                    try:
                        # Add all the GDCS genes to the list
                        for gene in sorted(sample[analysistype].faidict):
                            if gene not in gdcs:
                                gdcs.append(gene)
                    except AttributeError:
                        sample[analysistype].createreport = False
                else:
                    sample[analysistype].createreport = False
            else:
                sample[analysistype].createreport = False
                sample.general.incomplete = True
        header = 'Strain,Genus,Matches,MeanCoverage,Pass/Fail,{},\n'.format(','.join(gdcs))
        data = str()
        with open(os.path.join(self.reportpath, '{}.csv'.format(analysistype)), 'w') as report:
            # Sort the samples in the report based on the closest refseq genus e.g. all samples with the same genus
            # will be grouped together in the report
            for genus in genera:
                for sample in self.runmetadata.samples:
                    if sample.general.closestrefseqgenus == genus:
                        if sample[analysistype].createreport:
                            sample[analysistype].totaldepth = list()
                            # Add the sample to the report if it matches the current genus
                            # if genus == sample.general.closestrefseqgenus:
                            data += '{},{},'.format(sample.name, genus)
                            # Initialise a variable to store the number of GDCS genes were matched
                            count = 0
                            # As I want the count to be in the report before all the gene results, this string will
                            # store the specific sample information, and will be added to data once count is known
                            specific = str()
                            for gene in gdcs:
                                # As there are different genes present in the GDCS databases for each organism of
                                # interest, genes that did not match because they're absent in the specific database are
                                # indicated using an X
                                if gene not in [result for result in sample[analysistype].faidict]:
                                    specific += 'X,'
                                else:
                                    try:
                                        # Report the necessary information for each gene result
                                        identity = sample[analysistype].results[gene]
                                        specific += '{}% ({} +/- {}),'\
                                            .format(identity, sample[analysistype].avgdepth[gene],
                                                    sample[analysistype].standarddev[gene])
                                        sample[analysistype].totaldepth.append(
                                            float(sample[analysistype].avgdepth[gene]))
                                        count += 1
                                    # If the gene was missing from the results attribute, add a - to the cell
                                    except (KeyError, AttributeError):
                                        sample.general.incomplete = True
                                        specific += '-,'
                            # Calculate the mean depth of the genes and the standard deviation
                            sample[analysistype].mean = numpy.mean(sample[analysistype].totaldepth)
                            sample[analysistype].stddev = numpy.std(sample[analysistype].totaldepth)
                            # Determine whether the sample pass the necessary quality criteria:
                            # Pass, all GDCS, mean coverage greater than 20X coverage;
                            # ?: Indeterminate value;
                            # -: Fail value
                            # Allow one missing GDCS to still be considered a pass
                            if count >= len(sample[analysistype].faidict) - 1:
                                if sample[analysistype].mean > 20:
                                    quality = '+'
                                else:
                                    quality = '?'
                                    sample.general.incomplete = True
                            else:
                                quality = '-'
                                sample.general.incomplete = True
                            # Add the count, mean depth with standard deviation, the pass/fail determination,
                            #  and the total number of GDCS genes as well as the results
                            data += '{hits}/{total},{mean} +/- {std},{fail},{gdcs}\n'\
                                .format(hits=str(count),
                                        total=len(sample[analysistype].faidict),
                                        mean='{:.2f}'.format(sample[analysistype].mean),
                                        std='{:.2f}'.format(sample[analysistype].stddev),
                                        fail=quality,
                                        gdcs=specific)
                        # # Any samples with a best assembly of 'NA' are considered incomplete.
                        # else:
                        #     data += '{},{},,,-\n'.format(sample.name, sample.general.closestrefseqgenus)
                        #     sample.general.incomplete = True
                    elif sample.general.closestrefseqgenus == 'NA':
                        data += '{}\n'.format(sample.name)
                        sample.general.incomplete = True
            # Write the header and data to file
            report.write(header)
            report.write(data)