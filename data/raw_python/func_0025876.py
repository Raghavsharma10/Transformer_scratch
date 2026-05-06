def _filterCopy(self, dataFile, newStart, newEnd, newDataDir):
        """
        Copies the data file to a new file in the tmp dir, filtering it according to newStart and newEnd and adjusting 
        the times as appropriate so it starts from 0.
        :param dataFile: the input file.
        :param newStart: the new start time.
        :param newEnd: the new end time.
        :param newDataDir: the tmp dir to write to.
        :return: the device name & no of rows in the data.
        """
        import csv
        pathToData = os.path.split(dataFile)
        dataFileName = pathToData[1]
        dataDeviceName = os.path.split(pathToData[0])[1]
        os.makedirs(os.path.join(newDataDir, dataDeviceName), exist_ok=True)
        outputFile = os.path.join(newDataDir, dataDeviceName, dataFileName)
        dataCount = 0
        rowNum = 0
        with open(dataFile, mode='rt', newline='') as dataIn, open(outputFile, mode='wt', newline='') as dataOut:
            writer = csv.writer(dataOut, delimiter=',')
            for row in csv.reader(dataIn, delimiter=','):
                if len(row) > 0:
                    time = float(row[0])
                    if newStart <= time <= newEnd:
                        newRow = row[:]
                        if newStart > 0:
                            newRow[0] = "{0:.3f}".format(time - newStart)
                        writer.writerow(newRow)
                        dataCount += 1
                else:
                    logger.warning('Ignoring empty row ' + str(rowNum) + ' in ' + str(dataFile))
                rowNum += 1
        return dataDeviceName, dataCount