def _getTimeStamps(self, table):
        ''' get time stamps '''
        timeStamps = []
        for th in table.thead.tr.contents:
            if '\n' != th:
                timeStamps.append(th.getText())

        return timeStamps[1:]