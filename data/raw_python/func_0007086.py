def report(self):
        """
        Create the report for the user-supplied targets
        """
        # Add all the genes to the header
        header = 'Sample,'
        data = str()
        with open(os.path.join(self.reportpath, '{at}.csv'.format(at=self.analysistype)), 'w') as report:
            write_header = True
            for sample in self.runmetadata:
                data += sample.name + ','
                # Iterate through all the user-supplied target names
                for target in sorted(self.genes):
                    write_results = False
                    # There was an issue with 'target' not matching 'name' due to a dash being replaced by an underscore
                    # only in 'name'. This will hopefully address this issue
                    target = target.replace('-', '_')
                    if write_header:
                        header += '{target}_match_details,{target},'.format(target=target)
                    for name, identity in sample[self.analysistype].results.items():
                        # Ensure that all dashes are replaced with underscores
                        name = name.replace('-', '_')
                        # If the current target matches the target in the header, add the data to the string
                        if name == target:
                            write_results = True
                            gene_results = '{percent_id}% ({avgdepth} +/- {stddev}),{record},'\
                                .format(percent_id=identity,
                                        avgdepth=sample[self.analysistype].avgdepth[name],
                                        stddev=sample[self.analysistype].standarddev[name],
                                        record=sample[self.analysistype].sequences[target])
                            # Populate the data string appropriately
                            data += gene_results
                    # If the target is not present, write dashes to represent the results and sequence
                    if not write_results:
                        data += '-,-,'
                data += ' \n'
                write_header = False
            header += '\n'
            # Write the strings to the report
            report.write(header)
            report.write(data)