def datetime_entry(self, prompt, message=None, formats=['%x %X'], show_example=False,
            rofi_args=None, **kwargs):
        """Prompt the user to enter a date and time.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        formats: list of strings, optional
            The formats that the user can enter the date and time in. These
            should be format strings as accepted by the
            datetime.datetime.strptime() function from the standard library.
            They are tried in order, and the first that returns a datetime
            object without error is selected.  Note that the '%x %X' in the
            default list is the current locale's date and time representation.
        show_example: Boolean
            If True, the current date and time in the first format given is appended to
            the message.

        Returns
        -------
        datetime.datetime, or None if the dialog is cancelled.

        """
        def datetime_validator(text):
            # Try them in order.
            for format in formats:
                try:
                    dt = datetime.strptime(text, format)
                except ValueError:
                    continue
                else:
                    # This one worked; good enough for us.
                    return (dt, None)

            # None of the formats worked.
            return (None, 'Please enter a valid date and time.')

        # Add an example to the message?
        if show_example:
            message = message or ""
            message += "Current date and time in the correct format: " + datetime.now().strftime(formats[0])

        return self.generic_entry(prompt, datetime_validator, message, rofi_args, **kwargs)