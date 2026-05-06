def __writeData(self, targetPath, fields, rows):
        ''' write data '''
        if path.exists(targetPath):
            LOG.error("Target file exists: %s" % path.abspath(targetPath) )
            raise UfException(Errors.FILE_EXIST, "can't write to a existing file") #because xlwt doesn't support it

        with ExcelLib(fileName = targetPath, mode = ExcelLib.WRITE_MODE) as excel:
            excel.writeRow(0, fields)
            for index, row in enumerate(rows):
                excel.writeRow(index+1, row)