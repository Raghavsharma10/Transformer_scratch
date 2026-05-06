def trylock(self):
        "Try to acquire lock and return True; if cannot acquire the lock at this moment, return False."
        if self.locked:
            return True
        if self.lockroutine:
            return False
        waiter = self.scheduler.send(LockEvent(self.context, self.key, self))
        if waiter:
            return False
        else:
            self.locked = True
            return True