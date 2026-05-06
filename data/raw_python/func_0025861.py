def schedule(self, name, duration, startTime, description=None):
        """
        Schedules a new measurement with the given name.
        :param name:
        :param duration:
        :param startTime:
        :param description:
        :return: a tuple
            boolean: measurement was scheduled if true
            message: description, generally only used as an error code
        """
        if self._clashes(startTime, duration):
            return False, MEASUREMENT_TIMES_CLASH
        else:
            am = ActiveMeasurement(name, startTime, duration, self.targetStateProvider.state, description=description)
            logger.info("Scheduling measurement " + am.id + " for " + str(duration) + "s")
            self.activeMeasurements.append(am)
            devices = self.deviceController.scheduleMeasurement(am.id, am.duration, am.startTime)
            anyFail = False
            for device, status in devices.items():
                if status == 200:
                    deviceStatus = RecordStatus.SCHEDULED
                else:
                    deviceStatus = RecordStatus.FAILED
                    anyFail = True
                am.updateDeviceStatus(device.deviceId, deviceStatus)
            if anyFail:
                am.status = MeasurementStatus.FAILED
            else:
                if am.status is MeasurementStatus.NEW:
                    am.status = MeasurementStatus.SCHEDULED
            return True, None