def indexInfo(self, index):
        """ Returns information about a specific 
        planetary time. 
        
        """
        entry = self.table[index]
        info = {
            # Default is diurnal
            'mode': 'Day',
            'ruler': self.dayRuler(),
            'dayRuler': self.dayRuler(),
            'nightRuler': self.nightRuler(),
            'hourRuler': entry[2],
            'hourNumber': index + 1,
            'tableIndex': index,
            'start': Datetime.fromJD(entry[0], self.date.utcoffset),
            'end': Datetime.fromJD(entry[1], self.date.utcoffset)
        }
        if index >= 12:
            # Set information as nocturnal
            info.update({
                'mode': 'Night',
                'ruler': info['nightRuler'],
                'hourNumber': index + 1 - 12
            })
        return info