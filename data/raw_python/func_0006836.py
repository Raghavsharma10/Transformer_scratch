def targets(self):
        """
        Create the GenObject for the analysis type, create the hash file for baiting (if necessary)
        """
        for sample in self.runmetadata:
            if sample.general.bestassemblyfile != 'NA':
                setattr(sample, self.analysistype, GenObject())
                sample[self.analysistype].runanalysis = True
                sample[self.analysistype].targetpath = self.targetpath
                baitpath = os.path.join(self.targetpath, 'bait')
                sample[self.analysistype].baitfile = glob(os.path.join(baitpath, '*.fa'))[0]
                try:
                    sample[self.analysistype].outputdir = os.path.join(sample.run.outputdirectory, self.analysistype)
                except AttributeError:
                    sample[self.analysistype].outputdir = \
                        os.path.join(sample.general.outputdirectory, self.analysistype)
                    sample.run.outputdirectory = sample.general.outputdirectory
                sample[self.analysistype].logout = os.path.join(sample[self.analysistype].outputdir, 'logout.txt')
                sample[self.analysistype].logerr = os.path.join(sample[self.analysistype].outputdir, 'logerr.txt')
                sample[self.analysistype].baitedfastq = os.path.join(sample[self.analysistype].outputdir,
                                                                     '{}_targetMatches.fastq'.format(self.analysistype))
                sample[self.analysistype].complete = False