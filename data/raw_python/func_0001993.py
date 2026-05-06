def set_lastmodified_date(self, date=None):
        """
        Set the last modified date of a IOC to the current date.
        User may specify the date they want to set as well.

        :param date: Date value to set the last modified date to.  This should be in the xsdDate form.
         This defaults to the current date if it is not provided.
         xsdDate Form: YYYY-MM-DDTHH:MM:SS
        :return: True
        :raises: IOCParseError if date format is not valid.
        """
        if date:
            match = re.match(DATE_REGEX, date)
            if not match:
                raise IOCParseError('last-modified date is not valid.  Must be in the form YYYY-MM-DDTHH:MM:SS')
        ioc_et.set_root_lastmodified(self.root, date)
        return True