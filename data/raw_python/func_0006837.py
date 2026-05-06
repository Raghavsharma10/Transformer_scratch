def targets(self):
        """
        Using the data from the BLAST analyses, set the targets folder, and create the 'mapping file'. This is the
        genera-specific FASTA file that will be used for all the reference mapping; it replaces the 'bait file' in the
        code
        """
        logging.info('Performing analysis with {} targets folder'.format(self.analysistype))
        for sample in self.runmetadata:
            if sample.general.bestassemblyfile != 'NA':
                sample[self.analysistype].targetpath = \
                    os.path.join(self.targetpath, 'genera', sample[self.analysistype].genus, '')
                # There is a relatively strict databasing scheme necessary for the custom targets. Eventually,
                # there will be a helper script to combine individual files into a properly formatted combined file
                try:
                    sample[self.analysistype].mappingfile = glob('{}*.fa'
                                                                 .format(sample[self.analysistype].targetpath))[0]
                # If the fasta file is missing, raise a custom error
                except IndexError as e:
                    # noinspection PyPropertyAccess
                    e.args = ['Cannot find the combined fasta file in {}. Please note that the file must have a '
                              '.fasta extension'.format(sample[self.analysistype].targetpath)]
                    if os.path.isdir(sample[self.analysistype].targetpath):
                        raise
                    else:
                        sample.general.bestassemblyfile = 'NA'