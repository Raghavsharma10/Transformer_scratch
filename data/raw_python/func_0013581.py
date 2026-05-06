def writeExpression(self, rnaQuantificationId, quantfilename):
        """
        Reads the quantification results file and adds entries to the
        specified database.
        """
        isNormalized = self._isNormalized
        units = self._units
        with open(quantfilename, "r") as quantFile:
            quantificationReader = csv.reader(quantFile, delimiter=b"\t")
            header = next(quantificationReader)
            expressionLevelColNum = self.setColNum(
                                        header, self._expressionLevelCol)
            nameColNum = self.setColNum(header, self._nameCol)
            countColNum = self.setColNum(header, self._countCol, -1)
            confColLowNum = self.setColNum(header, self._confColLow, -1)
            confColHiNum = self.setColNum(header, self._confColHi, -1)
            expressionId = 0
            for expression in quantificationReader:
                expressionLevel = expression[expressionLevelColNum]
                name = expression[nameColNum]
                rawCount = 0.0
                if countColNum != -1:
                    rawCount = expression[countColNum]
                confidenceLow = 0.0
                confidenceHi = 0.0
                score = 0.0
                if confColLowNum != -1 and confColHiNum != -1:
                    confidenceLow = float(expression[confColLowNum])
                    confidenceHi = float(expression[confColHiNum])
                    score = (confidenceLow + confidenceHi)/2

                datafields = (expressionId, rnaQuantificationId, name,
                              expressionLevel, isNormalized, rawCount, score,
                              units, confidenceLow, confidenceHi)
                self._db.addExpression(datafields)
                expressionId += 1
            self._db.batchAddExpression()