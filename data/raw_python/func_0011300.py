def time_entry(self, prompt, message=None, formats=['%X', '%H:%M', '%I:%M', '%H.%M',
        '%I.%M'], show_example=False, rofi_args=None, **kwargs):
        """Prompt the user to enter a time.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        formats: list of strings, optional
            The formats that the user can enter times in. These should be
            format strings as accepted by the datetime.datetime.strptime()
            function from the standard library. They are tried in order, and
            the first that returns a time object without error is selected.
            Note that the '%X' in the default list is the current locale's time
            representation.
        show_example: Boolean
            If True, the current time in the first format given is appended to
            the message.

        Returns
        -------
        datetime.time, or None if the dialog is cancelled.

        """
        def time_validator(text):
            # Try them in order.
            for format in formats:
                try:
                    dt = datetime.strptime(text, format)
                except ValueError:
                    continue
                else:
                    # This one worked; good enough for us.
                    return (dt.time(), None)

            # None of the formats worked.
            return (None, 'Please enter a valid time.')

        # Add an example to the message?
        if show_example:
            message = message or ""
            message += "Current time in the correct format: " + datetime.now().strftime(formats[0])

        return self.generic_entry(prompt, time_validator, message, rofi_args=None, **kwargs)