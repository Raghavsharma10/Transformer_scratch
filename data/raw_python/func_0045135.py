def _get_horoscope_meta(self, day='today'):
        """gets a horoscope meta from site html

        :param day: day for which to get horoscope meta. Default is 'today'

        :returns: dictionary of horoscope mood details
        """
        if not is_valid_day(day):
            raise HoroscopeException("Invalid day. Allowed days: [today|yesterday|tomorrow]" )

        return {
            'intensity': str(self.tree.xpath('//*[@id="%s"]/div[3]/div[1]/p[1]/text()' % day)[0]).replace(": ", ""),
            'mood': str(self.tree.xpath('//*[@id="%s"]/div[3]/div[1]/p[2]/text()' % day)[0]).replace(": ", ""),
            'keywords': str(self.tree.xpath('//*[@id="%s"]/div[3]/div[2]/p[1]/text()' % day)[0]).replace(": ", ""),
        }