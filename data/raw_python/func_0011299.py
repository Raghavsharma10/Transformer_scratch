def date_entry(self, prompt, message=None, formats=['%x', '%d/%m/%Y'],
            show_example=False, rofi_args=None, **kwargs):
        """Prompt the user to enter a date.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        formats: list of strings, optional
            The formats that the user can enter dates in. These should be
            format strings as accepted by the datetime.datetime.strptime()
            function from the standard library. They are tried in order, and
            the first that returns a date object without error is selected.
            Note that the '%x' in the default list is the current locale's date
            representation.
        show_example: Boolean
            If True, today's date in the first format given is appended to the
            message.

        Returns
        -------
        datetime.date, or None if the dialog is cancelled.

        """
        def date_validator(text):
            # Try them in order.
            for format in formats:
                try:
                    dt = datetime.strptime(text, format)
                except ValueError:
                    continue
                else:
                    # This one worked; good enough for us.
                    return (dt.date(), None)

            # None of the formats worked.
            return (None, 'Please enter a valid date.')

        # Add an example to the message?
        if show_example:
            message = message or ""
            message += "Today's date in the correct format: " + datetime.now().strftime(formats[0])

        return self.generic_entry(prompt, date_validator, message, rofi_args, **kwargs)