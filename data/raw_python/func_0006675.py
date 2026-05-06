def targets(self):
        """
        Search the targets folder for FASTA files, create the multi-FASTA file of all targets if necessary, and
        populate objects
        """
        logging.info('Performing analysis with {} targets folder'.format(self.analysistype))
        for sample in self.runmetadata:
            sample[self.analysistype].runanalysis = True
            sample[self.analysistype].targetpath = (os.path.join(self.targetpath,
                                                                 sample[self.analysistype].pointfindergenus))
            # There is a relatively strict databasing scheme necessary for the custom targets. Eventually,
            # there will be a helper script to combine individual files into a properly formatted combined file
            try:
                sample[self.analysistype].baitfile = glob(os.path.join(sample[self.analysistype].targetpath,
                                                                       '*.fasta'))[0]
            # If the fasta file is missing, raise a custom error
            except IndexError:
                # Combine any .tfa files in the directory into a combined targets .fasta file
                fsafiles = glob(os.path.join(sample[self.analysistype].targetpath, '*.fsa'))
                if fsafiles:
                    combinetargets(fsafiles, sample[self.analysistype].targetpath)
                try:
                    sample[self.analysistype].baitfile = glob(os.path.join(sample[self.analysistype].targetpath,
                                                                           '*.fasta'))[0]
                except IndexError as e:
                    # noinspection PyPropertyAccess
                    e.args = [
                        'Cannot find the combined fasta file in {}. Please note that the file must have a '
                        '.fasta extension'.format(sample[self.analysistype].targetpath)]
                    if os.path.isdir(sample[self.analysistype].targetpath):
                        raise
                    else:
                        sample[self.analysistype].runanalysis = False
        for sample in self.runmetadata:
            # Set the necessary attributes
            sample[self.analysistype].outputdir = os.path.join(sample.run.outputdirectory, self.analysistype)
            make_path(sample[self.analysistype].outputdir)
            sample[self.analysistype].logout = os.path.join(sample[self.analysistype].outputdir, 'logout.txt')
            sample[self.analysistype].logerr = os.path.join(sample[self.analysistype].outputdir, 'logerr.txt')
            sample[self.analysistype].baitedfastq = \
                os.path.join(sample[self.analysistype].outputdir,
                             '{at}_targetMatches.fastq.gz'.format(at=self.analysistype))