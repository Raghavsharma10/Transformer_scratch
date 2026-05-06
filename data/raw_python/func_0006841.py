def makeblastdb(self):
        """
        Makes blast database files from targets as necessary
        """
        # Iterate through the samples to set the bait file.
        for sample in self.runmetadata.samples:
            if sample.general.bestassemblyfile != 'NA':
                # Remove the file extension
                db = os.path.splitext(sample[self.analysistype].baitfile)[0]
                # Add '.nhr' for searching below
                nhr = '{}.nhr'.format(db)
                # Check for already existing database files
                if not os.path.isfile(str(nhr)):
                    # Create the databases
                    command = 'makeblastdb -in {} -parse_seqids -max_file_sz 2GB -dbtype nucl -out {}'\
                        .format(sample[self.analysistype].baitfile, db)
                    out, err = run_subprocess(command)
                    write_to_logfile(command,
                                     command,
                                     self.logfile, sample.general.logout, sample.general.logerr,
                                     sample[self.analysistype].logout, sample[self.analysistype].logerr)
                    write_to_logfile(out,
                                     err,
                                     self.logfile, sample.general.logout, sample.general.logerr,
                                     sample[self.analysistype].logout, sample[self.analysistype].logerr)