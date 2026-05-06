def move_mouse(self, x, y, screen=0):
        """
        Move the mouse to a specific location.

        :param x: the target X coordinate on the screen in pixels.
        :param y: the target Y coordinate on the screen in pixels.
        :param screen: the screen (number) you want to move on.
        """
        # todo: apparently the "screen" argument is not behaving properly
        #       and sometimes even making the interpreter crash..
        #       Figure out why (changed API / using wrong header?)

        # >>> xdo.move_mouse(3000,200,1)

        # X Error of failed request:  BadWindow (invalid Window parameter)
        #   Major opcode of failed request:  41 (X_WarpPointer)
        #   Resource id in failed request:  0x2a4fca0
        #   Serial number of failed request:  25
        #   Current serial number in output stream:  26

        # Just to be safe..
        # screen = 0

        x = ctypes.c_int(x)
        y = ctypes.c_int(y)
        screen = ctypes.c_int(screen)

        _libxdo.xdo_move_mouse(self._xdo, x, y, screen)