def load_module(self, path, init_func):
        '''Load a shared library.

        Call this function to load a shared library (DLL file under Windows,
        shared object under UNIX) into the manager.

        @param path The path to the shared library.
        @param init_func The name entry function in the library.
        @raises FailedToLoadModuleError

        '''
        try:
            with self._mutex:
                if self._obj.load_module(path, init_func) != RTC.RTC_OK:
                    raise exceptions.FailedToLoadModuleError(path)
        except CORBA.UNKNOWN as e:
            if e.args[0] == UNKNOWN_UserException:
                raise exceptions.FailedToLoadModuleError(path, 'CORBA User Exception')
            else:
                raise