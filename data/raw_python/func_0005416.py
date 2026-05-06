def _set_date_tuple(self, mediafile, year, month=None, day=None):
        """Set the value of the field given a year, month, and day
        number. Each number can be an integer or None to indicate an
        unset component.
        """
        if year is None:
            self.__delete__(mediafile)
            return

        date = [u'{0:04d}'.format(int(year))]
        if month:
            date.append(u'{0:02d}'.format(int(month)))
        if month and day:
            date.append(u'{0:02d}'.format(int(day)))
        date = map(six.text_type, date)
        super(DateField, self).__set__(mediafile, u'-'.join(date))

        if hasattr(self, '_year_field'):
            self._year_field.__set__(mediafile, year)