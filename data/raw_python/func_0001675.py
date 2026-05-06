def _run_grid_multiprocess(self, func, iterables):
        ''' running case with mutil process to support selenium grid-mode(multiple web) and appium grid-mode(multiple devices). 
        @param func:  function object
        @param iterables:  iterable objects
        '''
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool()        
        pool_tracers = pool.map(func, iterables)
        pool.close()
        pool.join()
        
        # 传递给 pool.map的 实例对象，内存地址发生变化， 因此，这里在运行结束后，重新定义 self.tracers 
        self.tracers = dict(zip(self._default_devices, pool_tracers))