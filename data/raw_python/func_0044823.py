def start(self, timeout=None):
        """
        uses multiprocess Process to call _wrapper_func in subprocess 
        """

        if self.is_alive():
            log.warning("a process with pid %s is already running", self._proc.pid)
            return
            
        self._run.value = True
        self._func_running.value = False
        name = self.__class__.__name__
        
        self.conn_recv, self.conn_send = mp.Pipe(False)
        self._monitor_thread = threading.Thread(target = self._monitor_stdout_pipe)
        self._monitor_thread.daemon=True
        self._monitor_thread.start()
        log.debug("started monitor thread")
        self._log_queue = mp.Queue()
        self._log_queue_listener = QueueListener(self._log_queue, *log.handlers)
        self._log_queue_listener.start()

        args = (self.func, self.args, self._run, self._pause, self.interval,
                self._sigint, self._sigterm, name, log.level, self.conn_send, 
                self._func_running, self._log_queue)
        
        self._proc = mp.Process(target = _loop_wrapper_func,
                                args   = args)
        self._proc.start()
        log.info("started a new process with pid %s", self._proc.pid)
        log.debug("wait for loop function to come up")
        t0 = time.time()
        while not self._func_running.value:
            if self._proc.exitcode is not None:
                exc = self._proc.exitcode
                self._proc = None
                if exc == 0:
                    log.warning("wrapper function already terminated with exitcode 0\nloop is not running")
                    return
                else:
                    raise LoopExceptionError("the loop function return non zero exticode ({})!\n".format(exc)+
                                             "see log (INFO level) for traceback information")
            
            time.sleep(0.1)           
            if (timeout is not None) and ((time.time() - t0) > timeout):
                err_msg = "could not bring up function on time (timeout: {}s)".format(timeout)
                log.error(err_msg)
                log.info("either it takes too long to spawn the subprocess (increase the timeout)\n"+
                         "or an internal error occurred before reaching the function call")
                raise LoopTimeoutError(err_msg)
            
        log.debug("loop function is up ({})".format(humanize_time(time.time()-t0)))