def get_formated_creation_date(self, format='%b %Y %d %H:%I:%S'):
        """ Return creation date with a given format. Default is '%b %Y %d %H:%I:%S' """

        if not self._creation_date:
            return None

        date = datetime.datetime.utcfromtimestamp(self._creation_date)
        return date.strftime(format)