def get_lock(self, lockname, locktime=60, auto_renewal=False):
        ''' Gets a lock and returns if it can be stablished. Returns false otherwise '''
        pid = os.getpid()
        caller = inspect.stack()[0][3]
        try:
            # rl = redlock.Redlock([{"host": settings.REDIS_SERVERS['std_redis']['host'], "port": settings.REDIS_SERVERS['std_redis']['port'], "db": settings.REDIS_SERVERS['std_redis']['db']}, ])
            rl = redis_lock.Lock(self, lockname, expire=locktime, auto_renewal=auto_renewal)
        except:
            if self.logger:
                self.logger.error('Process {0} ({1}) could not get lock {2}. Going ahead without locking!!! {3}'.format(pid, caller, lockname, traceback.format_exc()))
            return False
        try:
            lock = rl.acquire(blocking=False)
        except RedisError:
            return False
        if not lock:
            return False
        else:
            return rl