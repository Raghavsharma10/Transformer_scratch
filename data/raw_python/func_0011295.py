def text_entry(self, prompt, message=None, allow_blank=False, strip=True,
            rofi_args=None, **kwargs):
        """Prompt the user to enter a piece of text.

        Parameters
        ----------
        prompt: string
            Prompt to display to the user.
        message: string, optional
            Message to display under the entry line.
        allow_blank: Boolean
            Whether to allow blank entries.
        strip: Boolean
            Whether to strip leading and trailing whitespace from the entered
            value.

        Returns
        -------
        string, or None if the dialog was cancelled.

        """
        def text_validator(text):
            if strip:
                text = text.strip()
            if not allow_blank:
                if not text:
                    return None, "A value is required."

            return text, None

        return self.generic_entry(prompt, text_validator, message, rofi_args, **kwargs)