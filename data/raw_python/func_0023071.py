def get_window_id(self):
        """ Get the window id of a PySide Widget. Might also work for PyQt4.
        """
        # Get Qt win id
        winid = self.winId()

        # On Linux this is it
        if IS_RPI:
            nw = (ctypes.c_int * 3)(winid, self.width(), self.height())
            return ctypes.pointer(nw)
        elif IS_LINUX:
            return int(winid)  # Is int on PySide, but sip.voidptr on PyQt

        # Get window id from stupid capsule thingy
        # http://translate.google.com/translate?hl=en&sl=zh-CN&u=http://www.cnb
        #logs.com/Shiren-Y/archive/2011/04/06/2007288.html&prev=/search%3Fq%3Dp
        # yside%2Bdirectx%26client%3Dfirefox-a%26hs%3DIsJ%26rls%3Dorg.mozilla:n
        #l:official%26channel%3Dfflb%26biw%3D1366%26bih%3D614
        # Prepare
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object,
                                                          ctypes.c_char_p]
        # Extract handle from capsule thingy
        name = ctypes.pythonapi.PyCapsule_GetName(winid)
        handle = ctypes.pythonapi.PyCapsule_GetPointer(winid, name)
        return handle