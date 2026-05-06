def reporter(self):
        """
        Creates a report of the results
        """
        # Create the path in which the reports are stored
        make_path(self.reportpath)
        header = 'Strain,Gene,PercentIdentity,Genus,FoldCoverage\n'
        data = ''
        with open(os.path.join(self.reportpath, self.analysistype + '.csv'), 'w') as report:
            for sample in self.runmetadata.samples:
                data += sample.name + ','
                if sample[self.analysistype].results:
                    if not sample[self.analysistype].multiple:
                        for name, identity in sample[self.analysistype].results.items():
                            if name == sample[self.analysistype].besthit[0]:
                                data += '{},{},{},{}\n'.format(name, identity, sample[self.analysistype].genera[name],
                                                               sample[self.analysistype].avgdepth[name])
                    else:
                        data += '{},{},{},{}\n'.format('multiple', 'NA', ';'.join(sample[self.analysistype]
                                                                                  .classification), 'NA')
                else:
                    data += '\n'
            report.write(header)
            report.write(data)