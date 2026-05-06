def solarReturn(self, year):
        """ Returns this chart's solar return for a 
        given year. 
        
        """
        sun = self.getObject(const.SUN)
        date = Datetime('{0}/01/01'.format(year),
                        '00:00',
                        self.date.utcoffset)
        srDate = ephem.nextSolarReturn(date, sun.lon)
        return Chart(srDate, self.pos, hsys=self.hsys)