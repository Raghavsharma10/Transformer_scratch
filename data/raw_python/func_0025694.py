def start(self, measurementId, durationInSeconds=None):
        """
        Initialises the device if required then enters a read loop taking data from the provider and passing it to the
         handler. It will continue until either breakRead is true or the duration (if provided) has passed.
        :return:
        """
        logger.info(">> measurement " + measurementId +
                    ((" for " + str(durationInSeconds)) if durationInSeconds is not None else " until break"))
        self.failureCode = None
        self.measurementOverflowed = False
        self.dataHandler.start(measurementId)
        self.breakRead = False
        self.startTime = time.time()
        self.doInit()
        # this must follow doInit because doInit sets status to INITIALISED
        self.status = RecordingDeviceStatus.RECORDING
        elapsedTime = 0
        try:
            self._sampleIdx = 0
            while True:
                logger.debug(measurementId + " provideData ")
                self.dataHandler.handle(self.provideData())
                elapsedTime = time.time() - self.startTime
                if self.breakRead or durationInSeconds is not None and elapsedTime > durationInSeconds:
                    logger.debug(measurementId + " breaking provideData")
                    self.startTime = 0
                    break
        except:
            self.status = RecordingDeviceStatus.FAILED
            self.failureCode = str(sys.exc_info())
            logger.exception(measurementId + " failed")
        finally:
            expectedSamples = self.fs * (durationInSeconds if durationInSeconds is not None else elapsedTime)
            if self._sampleIdx < expectedSamples:
                self.status = RecordingDeviceStatus.FAILED
                self.failureCode = "Insufficient samples " + str(self._sampleIdx) + " for " + \
                                   str(elapsedTime) + " second long run, expected " + str(expectedSamples)
            self._sampleIdx = 0
            if self.measurementOverflowed:
                self.status = RecordingDeviceStatus.FAILED
                self.failureCode = "Measurement overflow detected"
            if self.status == RecordingDeviceStatus.FAILED:
                logger.error("<< measurement " + measurementId + " - FAILED - " + self.failureCode)
            else:
                self.status = RecordingDeviceStatus.INITIALISED
                logger.info("<< measurement " + measurementId + " - " + self.status.name)
            self.dataHandler.stop(measurementId, self.failureCode)
            if self.status == RecordingDeviceStatus.FAILED:
                logger.warning("Reinitialising device after measurement failure")
                self.doInit()