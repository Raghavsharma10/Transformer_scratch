def float_entry(self, prompt, message=None, min=None, max=None, rofi_args=None, **kwargs):
        """Prompt the user to enter a floating point number.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        min, max: float, optional
            Minimum and maximum values to allow. If None, no limit is imposed.

        Returns
        -------
        float, or None if the dialog is cancelled.

        """
        # Sanity check.
        if (min is not None) and (max is not None) and not (max > min):
            raise ValueError("Maximum limit has to be more than the minimum limit.")

        def float_validator(text):
            error = None

            # Attempt to convert to float.
            try:
                value = float(text)
            except ValueError:
                return None, "Please enter a floating point value."

            # Check its within limits.
            if (min is not None) and (value < min):
                return None, "The minimum allowable value is {0}.".format(min)
            if (max is not None) and (value > max):
                return None, "The maximum allowable value is {0}.".format(max)

            return value, None

        return self.generic_entry(prompt, float_validator, message, rofi_args, **kwargs)