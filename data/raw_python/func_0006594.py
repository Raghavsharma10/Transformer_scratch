def main(self):
        """
        Run the necessary methods in the correct order
        """
        logging.info('Starting {at} analysis pipeline'.format(at=self.analysistype))
        # Create the objects to be used in the analyses
        objects = Objectprep(self)
        objects.objectprep()
        self.runmetadata = objects.samples
        self.threads = int(self.cpus / len(self.runmetadata.samples)) if self.cpus / len(self.runmetadata.samples) > 1 \
            else 1
        if self.genesippr:
            # Run the genesippr analyses
            self.analysistype = 'genesippr'
            self.targetpath = os.path.join(self.reffilepath, self.analysistype)
            Sippr(inputobject=self,
                  cutoff=0.90,
                  averagedepth=5)
            # Create the reports
            self.reports = Reports(self)
            Reports.reporter(self.reports)
        if self.sixteens:
            # Run the 16S analyses
            SixteensFull(args=self,
                         pipelinecommit=self.commit,
                         startingtime=self.starttime,
                         scriptpath=self.homepath,
                         analysistype='sixteens_full',
                         cutoff=0.985)
        if self.closestreference:
            self.pipeline = True
            mash.Mash(inputobject=self,
                      analysistype='mash')
        if self.rmlst:
            rmlst = MLSTSippr(args=self,
                              pipelinecommit=self.commit,
                              startingtime=self.starttime,
                              scriptpath=self.homepath,
                              analysistype='rMLST',
                              cutoff=1.0,
                              pipeline=True)
            rmlst.runner()
        if self.resistance:
            # ResFinding
            res = Resistance(args=self,
                             pipelinecommit=self.commit,
                             startingtime=self.starttime,
                             scriptpath=self.homepath,
                             analysistype='resfinder',
                             cutoff=0.7,
                             pipeline=False,
                             revbait=True)
            res.main()
        if self.virulence:
            self.genus_specific()
            Virulence(args=self,
                      pipelinecommit=self.commit,
                      startingtime=self.starttime,
                      scriptpath=self.homepath,
                      analysistype='virulence',
                      cutoff=0.95,
                      pipeline=False,
                      revbait=True)
        if self.gdcs:
            self.genus_specific()
            # Run the GDCS analysis
            self.analysistype = 'GDCS'
            self.targetpath = os.path.join(self.reffilepath, self.analysistype)
            Sippr(inputobject=self,
                  cutoff=0.95,
                  k=self.gdcs_kmer_size)
            # Create the reports
            self.reports = Reports(self)
            Reports.gdcsreporter(self.reports)
        if self.mlst:
            self.genus_specific()
            mlst = MLSTSippr(args=self,
                             pipelinecommit=self.commit,
                             startingtime=self.starttime,
                             scriptpath=self.homepath,
                             analysistype='MLST',
                             cutoff=1.0,
                             pipeline=True)
            mlst.runner()
        # Serotyping
        if self.serotype:
            self.genus_specific()
            SeroSippr(args=self,
                      pipelinecommit=self.commit,
                      startingtime=self.starttime,
                      scriptpath=self.homepath,
                      analysistype='serosippr',
                      cutoff=0.90,
                      pipeline=True)
        # Point mutation detection
        if self.pointfinder:
            self.genus_specific()
            PointSippr(args=self,
                       pipelinecommit=self.commit,
                       startingtime=self.starttime,
                       scriptpath=self.homepath,
                       analysistype='pointfinder',
                       cutoff=0.85,
                       pipeline=True,
                       revbait=True)
        if self.user_genes:
            custom = CustomGenes(args=self,
                                 kmer_size=self.kmer_size,
                                 allow_soft_clips=self.allow_soft_clips)
            custom.main()
        # Print the metadata
        MetadataPrinter(self)