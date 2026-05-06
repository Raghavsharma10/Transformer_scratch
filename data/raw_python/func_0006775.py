def main(self):
        """
        Run the methods required to create the genesippr report summary image
        """
        self.dataframe_setup()
        self.figure_populate(self.outputfolder,
                             self.image_report,
                             self.header_list,
                             self.samples,
                             'genesippr',
                             'report',
                             fail=self.fail)