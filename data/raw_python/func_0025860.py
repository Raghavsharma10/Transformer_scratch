def _sweep(self):
        """
        Checks the state of each measurement and verifies their state, if an active measurement is now complete then
        passes them to the completed measurement set, if failed then to the failed set, if failed and old then evicts.
        :return:
        """
        while self.running:
            for am in list(self.activeMeasurements):
                now = datetime.datetime.utcnow()
                # devices were allocated and have completed == complete
                recordingDeviceCount = len(am.recordingDevices)
                if recordingDeviceCount > 0:
                    if all(entry['state'] == RecordStatus.COMPLETE.name for entry in am.recordingDevices.values()):
                        logger.info("Detected completedmeasurement " + am.id)
                        self._moveToComplete(am)

                # we have reached the end time and we have either all failed devices or no devices == kill
                if now > (am.endTime + datetime.timedelta(days=0, seconds=1)):
                    allFailed = all(entry['state'] == RecordStatus.FAILED.name
                                    for entry in am.recordingDevices.values())
                    if (recordingDeviceCount > 0 and allFailed) or recordingDeviceCount == 0:
                        logger.warning("Detected failed measurement " + am.id + " with " + str(recordingDeviceCount)
                                       + " devices, allFailed: " + str(allFailed))
                        self._moveToFailed(am)

                # we are well past the end time and we have failed devices or an ongoing recording == kill or deathbed
                if now > (am.endTime + datetime.timedelta(days=0, seconds=self.maxTimeTilDeathbedSeconds)):
                    if any(entry['state'] == RecordStatus.FAILED.name for entry in am.recordingDevices.values()):
                        logger.warning("Detected failed and incomplete measurement " + am.id + ", assumed dead")
                        self._moveToFailed(am)
                    elif all(entry['state'] == RecordStatus.RECORDING.name for entry in am.recordingDevices.values()):
                        self._handleDeathbed(am)
            time.sleep(0.1)
        logger.warning("MeasurementCaretaker is now shutdown")