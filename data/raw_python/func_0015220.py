def ask_for_password(cls, ui, prompt='Provide your password:', **options):
        """Returns the password typed by user as a string or None if user cancels the request
        (e.g. presses Ctrl + D on commandline or presses Cancel in GUI.
        """
        # optionally set title, that may be used by some helpers like zenity
        return cls.get_appropriate_helper(ui).ask_for_password(prompt,
                                                               title=options.get('title', prompt))