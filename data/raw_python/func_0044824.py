def stop(self):
        """
        stops the process triggered by start
        
        Setting the shared memory boolean run to false, which should prevent
        the loop from repeating. Call __cleanup to make sure the process
        stopped. After that we could trigger start() again.
        """        
        if self.is_alive():
            self._proc.terminate()
            
        if self._proc is not None:
            self.__cleanup()
                   
            if self.raise_error:
                if self._proc.exitcode == 255:
                    raise LoopExceptionError("the loop function return non zero exticode ({})!\n".format(self._proc.exitcode)+
                                             "see log (INFO level) for traceback information")
        self.pipe_handler.close()
        self._proc = None