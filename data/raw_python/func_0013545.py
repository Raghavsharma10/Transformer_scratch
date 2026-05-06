def getExperiment(self):
        """
        Returns the GA4GH protocol representation of this read group's
        Experiment.
        """
        experiment = protocol.Experiment()
        experiment.id = self.getExperimentId()
        experiment.instrument_model = pb.string(self.getInstrumentModel())
        experiment.sequencing_center = pb.string(self.getSequencingCenter())
        experiment.description = pb.string(self.getExperimentDescription())
        experiment.library = pb.string(self.getLibrary())
        experiment.platform_unit = pb.string(self.getPlatformUnit())
        experiment.message_create_time = self._iso8601
        experiment.message_update_time = self._iso8601
        experiment.run_time = pb.string(self.getRunTime())
        return experiment