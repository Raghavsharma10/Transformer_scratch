def set_created_date(self, date=None):
        """
        Set the created date of a IOC to the current date.
        User may specify the date they want to set as well.

        :param date: Date value to set the created date to.  This should be in the xsdDate form.
         This defaults to the current date if it is not provided.
         xsdDate form: YYYY-MM-DDTHH:MM:SS
        :return: True
        :raises: IOCParseError if date format is not valid.
        """
        if date:
            match = re.match(DATE_REGEX, date)
            if not match:
                raise IOCParseError('Created date is not valid.  Must be in the form YYYY-MM-DDTHH:MM:SS')
        # XXX can this use self.metadata?
        ioc_et.set_root_created_date(self.root, date)
        return True