def click(self, force_no_call=False, milis=None):
        """
        Call when the button is pressed. This start the callback function in a thread
        If :milis is given, will release the button after :milis miliseconds
        """

        if self.clicked:
            return False

        if not force_no_call and self.flags & self.CALL_ON_PRESS:
            if self.flags & self.THREADED_CALL:
                start_new_thread(self.func, ())
            else:
                self.func()

        super().click()

        if milis is not None:
            start_new_thread(self.release, (), {'milis': milis})