def decimal_entry(self, prompt, message=None, min=None, max=None, rofi_args=None, **kwargs):
        """Prompt the user to enter a decimal number.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        min, max: Decimal, optional
            Minimum and maximum values to allow. If None, no limit is imposed.

        Returns
        -------
        Decimal, or None if the dialog is cancelled.

        """
        # Sanity check.
        if (min is not None) and (max is not None) and not (max > min):
            raise ValueError("Maximum limit has to be more than the minimum limit.")

        def decimal_validator(text):
            error = None

            # Attempt to convert to decimal.
            try:
                value = Decimal(text)
            except InvalidOperation:
                return None, "Please enter a decimal value."

            # Check its within limits.
            if (min is not None) and (value < min):
                return None, "The minimum allowable value is {0}.".format(min)
            if (max is not None) and (value > max):
                return None, "The maximum allowable value is {0}.".format(max)

            return value, None

        return self.generic_entry(prompt, decimal_validator, message, rofi_args, **kwargs)