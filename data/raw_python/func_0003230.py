def beginlock(self, container):
        "Start to acquire lock in another routine. Call trylock or lock later to acquire the lock. Call unlock to cancel the lock routine"
        if self.locked:
            return True
        if self.lockroutine:
            return False
        self.lockroutine = container.subroutine(self._lockroutine(container), False)
        return self.locked