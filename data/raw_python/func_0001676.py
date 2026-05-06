def _run_grid_multithread(self, func, iterables):
        ''' running case with mutil process to support selenium grid-mode(multiple web) and appium grid-mode(multiple devices). 
        @param func:  function object
        @param iterables:  iterable objects
        '''
        
        f = lambda x: threading.Thread(target = func,args = (x,))
        threads = map(f, iterables)
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
            thread.join()