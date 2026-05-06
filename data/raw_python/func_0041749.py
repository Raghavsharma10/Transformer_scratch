def release_lock(self, lock, force=False):
        ''' Frees a lock '''
        pid = os.getpid()
        caller = inspect.stack()[0][3]
        # try:
        #   rl = redlock.Redlock([{"host": settings.REDIS_SERVERS['std_redis']['host'], "port": settings.REDIS_SERVERS['std_redis']['port'], "db": settings.REDIS_SERVERS['std_redis']['db']}, ])
        # except:
        #   logger.error('Process {0} ({1}) could not release lock {2}'.format(pid, caller, lock.resource))
        #   return False
        if lock and lock._held:
            lock.release()
        if self.logger:
            self.logger.debug('Process {0} ({1}) released lock'.format(pid, caller))