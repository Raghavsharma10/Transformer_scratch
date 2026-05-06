def wait_for_lock(self, lockname, locktime=60, auto_renewal=False):
        ''' Gets a lock or waits until it is able to get it '''
        pid = os.getpid()
        caller = inspect.stack()[0][3]
        try:
            # rl = redlock.Redlock([{"host": settings.REDIS_SERVERS['std_redis']['host'], "port": settings.REDIS_SERVERS['std_redis']['port'], "db": settings.REDIS_SERVERS['std_redis']['db']}, ])
            rl = redis_lock.Lock(self, lockname, expire=locktime, auto_renewal=auto_renewal)
        except AssertionError:
            if self.logger:
                self.logger.error('Process {0} ({1}) could not get lock {2}. Going ahead without locking!!! {3}'.format(pid, caller, lockname, traceback.format_exc()))
            return False
        cont = 1
        t0 = time.time()
        lock = None
        while not lock:
            time.sleep(.05)
            cont += 1
            if cont % 20 == 0:
                if self.logger:
                    self.logger.debug('Process {0} ({1}) waiting for lock {2}. {3} seconds elapsed.'.format(pid, caller, lockname, time.time() - t0))
            # lock = rl.lock(lockname, locktime_ms)
            try:
                lock = rl.acquire()
            except RedisError:
                pass
        if self.logger:
            self.logger.debug('Process {0} ({1}) got lock {2} for {3} seconds'.format(pid, caller, lockname, locktime))
        return rl