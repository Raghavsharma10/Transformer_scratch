def _get_zip_filename(self):
        """Create a filename for zip file when :class:Transfer.compress is
        set to ``True``

        :rtype: str
        """

        date = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')
        zip_file = 'filemail_transfer_{date}.zip'.format(date=date)

        return zip_file