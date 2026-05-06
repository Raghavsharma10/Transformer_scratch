def next_timer_delta(self):
        "Returns a timevalue that the proactor will wait on."
        if self.timeouts and not self.active:
            now = getnow()
            timo = self.timeouts[0].timeout
            if now >= timo:
                #looks like we've exceded the time
                return 0
            else:
                return (timo - now)
        else:
            if self.active:
                return 0
            else:
                return None