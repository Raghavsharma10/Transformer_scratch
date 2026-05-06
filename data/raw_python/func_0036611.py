def unload_module(self, path):
        '''Unload a loaded shared library.

        Call this function to remove a shared library (e.g. a component) that
        was previously loaded.

        @param path The path to the shared library.
        @raises FailedToUnloadModuleError

        '''
        with self._mutex:
            if self._obj.unload_module(path) != RTC.RTC_OK:
                raise FailedToUnloadModuleError(path)