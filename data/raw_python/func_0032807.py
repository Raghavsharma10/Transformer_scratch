def group_num(self):
        """Current group number.

        :getter: Returns current group number
        :setter: Sets current group number
        :type: int
        """
        xkb_state = XkbStateRec()
        XkbGetState(self._display, XkbUseCoreKbd, byref(xkb_state))
        return xkb_state.group