def _is_locked(self):
        '''
        Checks to see if we are already pulling items from the queue
        '''
        if os.path.isfile(self._lck):
            try:
                import psutil
            except ImportError:
                return True #Lock file exists and no psutil
            #If psutil is imported
            with open(self._lck) as f:
                pid = f.read()
            return True if psutil.pid_exists(int(pid)) else False
        else:
            return False