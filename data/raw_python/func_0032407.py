def __readData(self, targetPath, start, end):
        ''' read data '''
        ret = []
        if not path.exists(targetPath):
            LOG.error("Target file doesn't exist: %s" % path.abspath(targetPath) )
            return ret

        with ExcelLib(fileName = targetPath, mode = ExcelLib.READ_MODE) as excel:
            low, high = self.__findRange(excel, start, end)

            for index in range(low, high + 1):
                ret.append(excel.readRow(index))

        return ret