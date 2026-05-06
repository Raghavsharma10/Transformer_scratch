def close_display(self):
        """Closes connection with X server and cleans up objects
        created on open_display().
        """
        if hasattr(self, "_keyboard_description") and self._keyboard_description:
            names_mask = XkbSymbolsNameMask | XkbGroupNamesMask
            XkbFreeNames(self._keyboard_description, names_mask, True)
            XkbFreeControls(self._keyboard_description, XkbAllControlsMask, True)
            XkbFreeClientMap(self._keyboard_description, 0, True)
            del self._keyboard_description

        if hasattr(self, "_display") and self._display:
            XCloseDisplay(self._display)
            del self._display