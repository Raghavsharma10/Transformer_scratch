def populateFromRow(self, readGroupRecord):
        """
        Populate the instance variables using the specified DB row.
        """
        self._sampleName = readGroupRecord.samplename
        self._biosampleId = readGroupRecord.biosampleid
        self._description = readGroupRecord.description
        self._predictedInsertSize = readGroupRecord.predictedinsertsize
        stats = protocol.fromJson(readGroupRecord.stats, protocol.ReadStats)
        self._numAlignedReads = stats.aligned_read_count
        self._numUnalignedReads = stats.unaligned_read_count
        experiment = protocol.fromJson(
            readGroupRecord.experiment, protocol.Experiment)
        self._instrumentModel = experiment.instrument_model
        self._sequencingCenter = experiment.sequencing_center
        self._experimentDescription = experiment.description
        self._library = experiment.library
        self._platformUnit = experiment.platform_unit
        self._runTime = experiment.run_time