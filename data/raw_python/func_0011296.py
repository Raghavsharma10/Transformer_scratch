def integer_entry(self, prompt, message=None, min=None, max=None, rofi_args=None, **kwargs):
        """Prompt the user to enter an integer.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        min, max: integer, optional
            Minimum and maximum values to allow. If None, no limit is imposed.

        Returns
        -------
        integer, or None if the dialog is cancelled.

        """
        # Sanity check.
        if (min is not None) and (max is not None) and not (max > min):
            raise ValueError("Maximum limit has to be more than the minimum limit.")

        def integer_validator(text):
            error = None

            # Attempt to convert to integer.
            try:
                value = int(text)
            except ValueError:
                return None, "Please enter an integer value."

            # Check its within limits.
            if (min is not None) and (value < min):
                return None, "The minimum allowable value is {0:d}.".format(min)
            if (max is not None) and (value > max):
                return None, "The maximum allowable value is {0:d}.".format(max)

            return value, None

        return self.generic_entry(prompt, integer_validator, message, rofi_args, **kwargs)