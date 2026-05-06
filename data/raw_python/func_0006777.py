def dataframe_setup(self):
        """
        Set-up a report to store the desired header: sanitized string combinations
        """
        # Initialise a dictionary to store the sanitized headers and strings
        genesippr_dict = dict()
        # Try to open all the reports - use pandas to extract the results from any report that exists
        try:
            sippr_matrix = pd.read_csv(os.path.join(self.reportpath, 'genesippr.csv'),
                                       delimiter=',', index_col=0).T.to_dict()
        except FileNotFoundError:
            sippr_matrix = dict()
        try:
            conf_matrix = pd.read_csv(os.path.join(self.reportpath, 'confindr_report.csv'),
                                      delimiter=',', index_col=0).T.to_dict()
        except FileNotFoundError:
            conf_matrix = dict()
        try:
            gdcs_matrix = pd.read_csv(os.path.join(self.reportpath, 'GDCS.csv'),
                                      delimiter=',', index_col=0).T.to_dict()
        except FileNotFoundError:
            gdcs_matrix = dict()
        # Populate the header:sanitized string dictionary with results from all strains
        for sample in self.metadata:
            genesippr_dict[sample.name] = dict()
            try:
                genesippr_dict[sample.name]['eae'] = self.data_sanitise(sippr_matrix[sample.name]['eae'])
            except KeyError:
                genesippr_dict[sample.name]['eae'] = 0
            try:
                genesippr_dict[sample.name]['hlyAEc'] = self.data_sanitise(sippr_matrix[sample.name]['hlyAEc'])
            except KeyError:
                genesippr_dict[sample.name]['hlyAEc'] = 0
            try:
                genesippr_dict[sample.name]['VT1'] = self.data_sanitise(sippr_matrix[sample.name]['VT1'])
            except KeyError:
                genesippr_dict[sample.name]['VT1'] = 0
            try:
                genesippr_dict[sample.name]['VT2'] = self.data_sanitise(sippr_matrix[sample.name]['VT2'])
            except KeyError:
                genesippr_dict[sample.name]['VT2'] = 0
            try:
                genesippr_dict[sample.name]['hlyALm'] = self.data_sanitise(sippr_matrix[sample.name]['hlyALm'])
            except KeyError:
                genesippr_dict[sample.name]['hlyALm'] = 0
            try:
                genesippr_dict[sample.name]['IGS'] = self.data_sanitise(sippr_matrix[sample.name]['IGS'])
            except KeyError:
                genesippr_dict[sample.name]['IGS'] = 0
            try:
                genesippr_dict[sample.name]['inlJ'] = self.data_sanitise(sippr_matrix[sample.name]['inlJ'])
            except KeyError:
                genesippr_dict[sample.name]['inlJ'] = 0
            try:
                genesippr_dict[sample.name]['invA'] = self.data_sanitise(sippr_matrix[sample.name]['invA'])
            except KeyError:
                genesippr_dict[sample.name]['invA'] = 0
            try:
                genesippr_dict[sample.name]['stn'] = self.data_sanitise(sippr_matrix[sample.name]['stn'])
            except KeyError:
                genesippr_dict[sample.name]['stn'] = 0
            try:
                genesippr_dict[sample.name]['GDCS'] = self.data_sanitise(gdcs_matrix[sample.name]['Pass/Fail'],
                                                                         header='Pass/Fail')
            except KeyError:
                genesippr_dict[sample.name]['GDCS'] = 0
            try:
                genesippr_dict[sample.name]['Contamination'] = self.data_sanitise(
                    conf_matrix[sample.name]['ContamStatus'], header='ContamStatus')
            except KeyError:
                genesippr_dict[sample.name]['Contamination'] = 0
            try:
                genesippr_dict[sample.name]['Coverage'] = self.data_sanitise(
                    gdcs_matrix[sample.name]['MeanCoverage'], header='MeanCoverage')
            except KeyError:
                genesippr_dict[sample.name]['Coverage'] = 0
        # Create a report from the header: sanitized string dictionary to be used in the creation of the report image
        with open(self.image_report, 'w') as csv:
            data = '{}\n'.format(','.join(self.header_list))
            for strain in sorted(genesippr_dict):
                data += '{str},'.format(str=strain)
                for header in self.header_list[1:]:
                    data += '{value},'.format(value=genesippr_dict[strain][header])

                data = data.rstrip(',')
                data += '\n'
            csv.write(data)