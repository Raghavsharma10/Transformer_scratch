def confindr_reporter(self, analysistype='confindr'):
        """
        Creates a final report of all the ConFindr results
        """
        # Initialise the data strings
        data = 'Strain,Genus,NumContamSNVs,ContamStatus,PercentContam,PercentContamSTD\n'
        with open(os.path.join(self.reportpath, analysistype + '.csv'), 'w') as report:
            # Iterate through all the results
            for sample in self.runmetadata.samples:
                data += '{str},{genus},{numcontamsnv},{status},{pc},{pcs}\n'.format(
                    str=sample.name,
                    genus=sample.confindr.genus,
                    numcontamsnv=sample.confindr.num_contaminated_snvs,
                    status=sample.confindr.contam_status,
                    pc=sample.confindr.percent_contam,
                    pcs=sample.confindr.percent_contam_std
                )
            # Write the string to the report
            report.write(data)