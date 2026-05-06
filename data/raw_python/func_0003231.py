def unlock(self):
        "Unlock the key"
        if self.lockroutine:
            self.lockroutine.close()
            self.lockroutine = None
        if self.locked:
            self.locked = False
            self.scheduler.ignore(LockEvent.createMatcher(self.context, self.key, self))