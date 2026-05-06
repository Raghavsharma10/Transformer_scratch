def select(self, prompt, options, rofi_args=None, message="", select=None, **kwargs):
        """Show a list of options and return user selection.

        This method blocks until the user makes their choice.

        Parameters
        ----------
        prompt: string
            The prompt telling the user what they are selecting.
        options: list of strings
            The options they can choose from. Any newline characters are
            replaced with spaces.
        message: string, optional
            Message to show between the prompt and the options. This can
            contain Pango markup, and any text content should be escaped.
        select: integer, optional
            Set which option is initially selected.
        keyN: tuple (string, string); optional
            Custom key bindings where N is one or greater. The first entry in
            the tuple should be a string defining the key, e.g., "Alt+x" or
            "Delete".  Note that letter keys should be lowercase ie.e., Alt+a
            not Alt+A.

            The second entry should be a short string stating the action the
            key will take. This is displayed to the user at the top of the
            dialog. If None or an empty string, it is not displayed (but the
            binding is still set).

            By default, key1 through key9 are set to ("Alt+1", None) through
            ("Alt+9", None) respectively.

        Returns
        -------
        tuple (index, key)
            The index of the option the user selected, or -1 if they cancelled
            the dialog.
            Key indicates which key was pressed, with 0 being 'OK' (generally
            Enter), -1 being 'Cancel' (generally escape), and N being custom
            key N.

        """
        rofi_args = rofi_args or []
        # Replace newlines and turn the options into a single string.
        optionstr = '\n'.join(option.replace('\n', ' ') for option in options)

        # Set up arguments.
        args = ['rofi', '-dmenu', '-p', prompt, '-format', 'i']
        if select is not None:
            args.extend(['-selected-row', str(select)])

        # Key bindings to display.
        display_bindings = []

        # Configure the key bindings.
        user_keys = set()
        for k, v in kwargs.items():
            # See if the keyword name matches the needed format.
            if not k.startswith('key'):
                continue
            try:
                keynum = int(k[3:])
            except ValueError:
                continue

            # Add it to the set.
            key, action = v
            user_keys.add(keynum)
            args.extend(['-kb-custom-{0:s}'.format(k[3:]), key])
            if action:
                display_bindings.append("<b>{0:s}</b>: {1:s}".format(key, action))

        # And the global exit bindings.
        exit_keys = set()
        next_key = 10
        for key in self.exit_hotkeys:
            while next_key in user_keys:
                next_key += 1
            exit_keys.add(next_key)
            args.extend(['-kb-custom-{0:d}'.format(next_key), key])
            next_key += 1

        # Add any displayed key bindings to the message.
        message = message or ""
        if display_bindings:
            message += "\n" + "  ".join(display_bindings)
        message = message.strip()

        # If we have a message, add it to the arguments.
        if message:
            args.extend(['-mesg', message])

        # Add in common arguments.
        args.extend(self._common_args(**kwargs))
        args.extend(rofi_args)

        # Run the dialog.
        returncode, stdout = self._run_blocking(args, input=optionstr)

        # Figure out which option was selected.
        stdout = stdout.strip()
        index = int(stdout) if stdout else -1

        # And map the return code to a key.
        if returncode == 0:
            key = 0
        elif returncode == 1:
            key = -1
        elif returncode > 9:
            key = returncode - 9
            if key in exit_keys:
                raise SystemExit()
        else:
            self.exit_with_error("Unexpected rofi returncode {0:d}.".format(results.returncode))

        # And return.
        return index, key