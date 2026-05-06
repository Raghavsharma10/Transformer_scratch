def open_display(self):
        """Establishes connection with X server and prepares objects
        necessary to retrieve and send data.
        """
        self.close_display()    # Properly finish previous open_display()

        XkbIgnoreExtension(False)

        display_name = None
        major = c_int(XkbMajorVersion)
        minor = c_int(XkbMinorVersion)
        reason = c_int()

        self._display = XkbOpenDisplay(
            display_name,
            None, None, byref(major), byref(minor), byref(reason))
        if not self._display:
            if reason.value in OPEN_DISPLAY_ERRORS:
                # Assume POSIX conformance
                display_name = os.getenv("DISPLAY") or "default"

                raise X11Error(OPEN_DISPLAY_ERRORS[reason.value].format(
                    libname="xkbgroup",
                    used_major=XkbMajorVersion,
                    used_minor=XkbMinorVersion,
                    found_major=major.value,
                    found_minor=minor.value,
                    display_name=display_name)
                        + ".")
            else:
                raise X11Error("Unknown error {} from XkbOpenDisplay.".format(reason.value))

        self._keyboard_description = XkbGetMap(self._display, 0, XkbUseCoreKbd)
        if not self._keyboard_description:
            self.close_display()
            raise X11Error("Failed to get keyboard description.")

        # Controls mask doesn't affect the availability of xkb->ctrls->num_groups anyway
        # Just use a valid value, and xkb->ctrls->num_groups will be definitely set
        status = XkbGetControls(self._display, XkbAllControlsMask, self._keyboard_description)
        if status != Success:
            self.close_display()
            raise X11Error(GET_CONTROLS_ERRORS[status] + ".")

        names_mask = XkbSymbolsNameMask | XkbGroupNamesMask
        status = XkbGetNames(self._display, names_mask, self._keyboard_description)
        if status != Success:
            self.close_display()
            raise X11Error(GET_NAMES_ERRORS[status] + ".")