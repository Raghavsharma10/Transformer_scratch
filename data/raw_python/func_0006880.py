def reporter(self):
        """
        Create the MASH report
        """
        logging.info('Creating {} report'.format(self.analysistype))
        make_path(self.reportpath)
        header = 'Strain,ReferenceGenus,ReferenceFile,ReferenceGenomeMashDistance,Pvalue,NumMatchingHashes\n'
        data = ''
        for sample in self.metadata:
            try:
                data += '{},{},{},{},{},{}\n'.format(sample.name,
                                                     sample[self.analysistype].closestrefseqgenus,
                                                     sample[self.analysistype].closestrefseq,
                                                     sample[self.analysistype].mashdistance,
                                                     sample[self.analysistype].pvalue,
                                                     sample[self.analysistype].nummatches)
            except AttributeError:
                data += '{}\n'.format(sample.name)
        # Create the report file
        reportfile = os.path.join(self.reportpath, 'mash.csv')
        with open(reportfile, 'w') as report:
            report.write(header)
            report.write(data)