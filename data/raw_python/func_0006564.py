def methods(self):
        """
        Run the typing methods
        """
        self.contamination_detection()
        ReportImage(self, 'confindr')
        self.run_genesippr()
        ReportImage(self, 'genesippr')
        self.run_sixteens()
        self.run_mash()
        self.run_gdcs()
        ReportImage(self, 'gdcs')