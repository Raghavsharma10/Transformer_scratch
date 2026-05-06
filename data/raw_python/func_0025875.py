def editMeasurement(self, measurementId, data):
        """
        Edits the specified measurement with the provided data.
        :param measurementId: the measurement id.
        :param data: the data to update.  
        :return: true if the measurement was edited
        """
        oldMeasurement = self.getMeasurement(measurementId, measurementStatus=MeasurementStatus.COMPLETE)
        if oldMeasurement:
            import copy
            newMeasurement = copy.deepcopy(oldMeasurement)
            deleteOld = False
            createdFilteredCopy = False
            newName = data.get('name', None)
            newDesc = data.get('description', None)
            newStart = float(data.get('start', 0))
            newEnd = float(data.get('end', oldMeasurement.duration))
            newDuration = newEnd - newStart
            newDevices = data.get('devices', None)
            if newName:
                logger.info('Updating name from ' + oldMeasurement.name + ' to ' + newName)
                newMeasurement.updateName(newName)
                createdFilteredCopy = True
                deleteOld = True
            if newDesc:
                logger.info('Updating description from ' + str(oldMeasurement.description) + ' to ' + str(newDesc))
                newMeasurement.description = newDesc
            if newDuration != oldMeasurement.duration:
                logger.info('Copying measurement to allow support new duration ' + str(newDuration))
                if oldMeasurement.name == newMeasurement.name:
                    newMeasurement.updateName(newMeasurement.name + '-' + str(int(time.time())))
                newMeasurement.duration = newDuration
                createdFilteredCopy = True
            if createdFilteredCopy:
                logger.info('Copying measurement data from ' + oldMeasurement.idAsPath + ' to ' + newMeasurement.idAsPath)
                newMeasurementPath = self._getPathToMeasurementMetaDir(newMeasurement.idAsPath)
                dataSearchPattern = self._getPathToMeasurementMetaDir(oldMeasurement.idAsPath) + '/**/data.out'
                newDataCountsByDevice = [self._filterCopy(dataFile, newStart, newEnd, newMeasurementPath)
                                         for dataFile in glob.glob(dataSearchPattern)]
                for device, count in newDataCountsByDevice:
                    newMeasurement.recordingDevices.get(device)['count'] = count
            self.store(newMeasurement)
            if newDevices:
                for renames in newDevices:
                    logger.info('Updating device name from ' + str(renames[0]) + ' to ' + str(renames[1]))
                    deviceState = newMeasurement.recordingDevices.get(renames[0])
                    newMeasurement.recordingDevices[renames[1]] = deviceState
                    del newMeasurement.recordingDevices[renames[0]]
                    os.rename(os.path.join(self._getPathToMeasurementMetaDir(newMeasurement.idAsPath), renames[0]),
                              os.path.join(self._getPathToMeasurementMetaDir(newMeasurement.idAsPath), renames[1]))
                self.store(newMeasurement)
            if deleteOld or createdFilteredCopy or newDevices:
                self.completeMeasurements.append(newMeasurement)
            if deleteOld:
                self.delete(oldMeasurement.id)
            return True
        else:
            return False