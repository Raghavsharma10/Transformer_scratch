def __findRange(self, excelLib, start, end):
        ''' return low and high as excel range '''
        inc = 1
        low = 0
        high = 0
        dates = excelLib.readCol(0, 1)

        for index, date in enumerate(dates):
            if int(start) <= int(date):
                low = index + inc
                break

        if low:
            for index, date in reversed(list(enumerate(dates))):
                if int(date) <= int(end):
                    high = index + inc
                    break

        return low, high