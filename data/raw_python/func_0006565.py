def contamination_detection(self):
        """
        Calculate the levels of contamination in the reads
        """
        self.qualityobject = quality.Quality(self)
        self.qualityobject.contamination_finder(input_path=self.sequencepath,
                                                report_path=self.reportpath)