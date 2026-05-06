def _get_horoscope(self, day='today'):
        """gets a horoscope from site html

        :param day: day for which to get horoscope. Default is 'today'

        :returns: dictionary of horoscope details
        """
        if not is_valid_day(day):
            raise HoroscopeException("Invalid day. Allowed days: [today|yesterday|tomorrow]" )

        horoscope = ''.join([str(s).strip() for s in self.tree.xpath('//*[@id="%s"]/p/text()' % day)])

        if day is 'yesterday':
            date = self.date_today - timedelta(days=1)
        elif day is 'today':
            date = self.date_today
        elif day is 'tomorrow':
            date = self.date_today + timedelta(days=1)

        return {
            'date': date.strftime("%Y-%m-%d"),
            'sunsign': self.sunsign.capitalize(),
            'horoscope': horoscope + "(c) Kelli Fox, The Astrologer, http://new.theastrologer.com",
            'meta': self._get_horoscope_meta(day),
            'credit': '(c) Kelli Fox, The Astrologer, http://new.theastrologer.com'
        }