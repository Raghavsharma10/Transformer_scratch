def parse_pointfinder(self):
        """
        Create summary reports for the PointFinder outputs
        """
        # Create the nested dictionary that stores the necessary values for creating summary reports
        self.populate_summary_dict()
        # Clear out any previous reports
        for organism in self.summary_dict:
            for report in self.summary_dict[organism]:
                try:
                    os.remove(self.summary_dict[organism][report]['summary'])
                except FileNotFoundError:
                    pass
        for sample in self.runmetadata.samples:
            # Find the PointFinder outputs. If the outputs don't exist, create the appropriate entries in the
            # summary dictionary as required
            try:
                self.summary_dict[sample.general.referencegenus]['prediction']['output'] = \
                    glob(os.path.join(sample[self.analysistype].pointfinder_outputs, '{seq}*prediction.txt'
                                      .format(seq=sample.name)))[0]
            except IndexError:
                try:
                    self.summary_dict[sample.general.referencegenus]['prediction']['output'] = str()
                except KeyError:
                    self.populate_summary_dict(genus=sample.general.referencegenus,
                                               key='prediction')
            try:
                self.summary_dict[sample.general.referencegenus]['table']['output'] = \
                    glob(os.path.join(sample[self.analysistype].pointfinder_outputs, '{seq}*table.txt'
                                      .format(seq=sample.name)))[0]
            except IndexError:
                try:
                    self.summary_dict[sample.general.referencegenus]['table']['output'] = str()
                except KeyError:
                    self.populate_summary_dict(genus=sample.general.referencegenus,
                                               key='table')
            try:
                self.summary_dict[sample.general.referencegenus]['results']['output'] = \
                    glob(os.path.join(sample[self.analysistype].pointfinder_outputs, '{seq}*results.tsv'
                                      .format(seq=sample.name)))[0]
            except IndexError:
                try:
                    self.summary_dict[sample.general.referencegenus]['results']['output'] = str()
                except KeyError:
                    self.populate_summary_dict(genus=sample.general.referencegenus,
                                               key='results')
            # Process the predictions
            self.write_report(summary_dict=self.summary_dict,
                              seqid=sample.name,
                              genus=sample.general.referencegenus,
                              key='prediction')
            # Process the results summary
            self.write_report(summary_dict=self.summary_dict,
                              seqid=sample.name,
                              genus=sample.general.referencegenus,
                              key='results')

            # Process the table summary
            self.write_table_report(summary_dict=self.summary_dict,
                                    seqid=sample.name,
                                    genus=sample.general.referencegenus)