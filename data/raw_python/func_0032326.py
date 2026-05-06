def _parseTarget(self, target, keyTimeValue):
        ''' parse table for get financial '''
        table = target.table
        timestamps = self._getTimeStamps(table)

        for tr in table.tbody.findChildren('tr'):
            for i, td in enumerate(tr.findChildren('td')):
                if 0 == i:
                    key = td.getText()
                    if key not in keyTimeValue:
                        keyTimeValue[key] = {}
                else:
                    keyTimeValue[key][timestamps[i - 1]] = self._getValue(td)