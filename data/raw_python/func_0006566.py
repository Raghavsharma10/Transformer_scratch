def run_genesippr(self):
        """
        Run the genesippr analyses
        """
        GeneSippr(args=self,
                  pipelinecommit=self.commit,
                  startingtime=self.starttime,
                  scriptpath=self.homepath,
                  analysistype='genesippr',
                  cutoff=0.95,
                  pipeline=False,
                  revbait=False)