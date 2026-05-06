def reloadCompletedMeasurements(self):
        """
        Reloads the completed measurements from the backing store.
        """
        from pathlib import Path
        reloaded = [self.load(x.resolve()) for x in Path(self.dataDir).glob('*/*/*') if x.is_dir()]
        logger.info('Reloaded ' + str(len(reloaded)) + ' completed measurements')
        self.completeMeasurements = [x for x in reloaded if x is not None and x.status == MeasurementStatus.COMPLETE]
        self.failedMeasurements = [x for x in reloaded if x is not None and x.status == MeasurementStatus.FAILED]