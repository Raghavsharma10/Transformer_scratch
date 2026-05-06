def report_parse(self):
        """
        If the pipeline has previously been run on these data, instead of reading through the results, parse the
        report instead
        """
        # Initialise lists
        report_strains = list()
        genus_list = list()
        if self.analysistype == 'mlst':
            for sample in self.runmetadata.samples:
                try:
                    genus_list.append(sample.general.referencegenus)
                except AttributeError:
                    sample.general.referencegenus = 'ND'
                    genus_list.append(sample.general.referencegenus)
        # Read in the report
        if self.analysistype == 'mlst':
            for genus in genus_list:
                try:
                    report_name = os.path.join(self.reportpath, '{at}_{genus}.csv'.format(at=self.analysistype,
                                                                                          genus=genus))
                    report_strains = self.report_read(report_strains=report_strains,
                                                      report_name=report_name)
                except FileNotFoundError:
                    report_name = self.report
                    report_strains = self.report_read(report_strains=report_strains,
                                                      report_name=report_name)
        else:
            report_name = self.report
            report_strains = self.report_read(report_strains=report_strains,
                                              report_name=report_name)
        # Populate strains not in the report with 'empty' GenObject with appropriate attributes
        for sample in self.runmetadata.samples:
            if sample.name not in report_strains:
                setattr(sample, self.analysistype, GenObject())
                sample[self.analysistype].sequencetype = 'ND'
                sample[self.analysistype].matches = 0
                sample[self.analysistype].results = dict()