def generic_entry(self, prompt, validator=None, message=None, rofi_args=None, **kwargs):
        """A generic entry box.

        Parameters
        ----------
        prompt: string
            Text prompt for the entry.
        validator: function, optional
            A function to validate and convert the value entered by the user.
            It should take one parameter, the string that the user entered, and
            return a tuple (value, error). The value should be the users entry
            converted to the appropriate Python type, or None if the entry was
            invalid. The error message should be a string telling the user what
            was wrong, or None if the entry was valid. The prompt will be
            re-displayed to the user (along with the error message) until they
            enter a valid value. If no validator is given, the text that the
            user entered is returned as-is.
        message: string
            Optional message to display under the entry.

        Returns
        -------
        The value returned by the validator, or None if the dialog was
        cancelled.

        Examples
        --------
        Enforce a minimum entry length:
        >>> r = Rofi()
        >>> validator = lambda s: (s, None) if len(s) > 6 else (None, "Too short")
        >>> r.generic_entry('Enter a 7-character or longer string: ', validator)

        """
        error = ""
        rofi_args = rofi_args or []

        # Keep going until we get something valid.
        while True:
            args = ['rofi', '-dmenu', '-p', prompt, '-format', 's']

            # Add any error to the given message.
            msg = message or ""
            if error:
                msg = '<span color="#FF0000" font_weight="bold">{0:s}</span>\n{1:s}'.format(error, msg)
                msg = msg.rstrip('\n')

            # If there is actually a message to show.
            if msg:
                args.extend(['-mesg', msg])

            # Add in common arguments.
            args.extend(self._common_args(**kwargs))
            args.extend(rofi_args)

            # Run it.
            returncode, stdout = self._run_blocking(args, input="")

            # Was the dialog cancelled?
            if returncode == 1:
                return None

            # Get rid of the trailing newline and check its validity.
            text = stdout.rstrip('\n')
            if validator:
                value, error = validator(text)
                if not error:
                    return value
            else:
                return text