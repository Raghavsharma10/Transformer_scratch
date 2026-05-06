def needs_distribute_ready(self):
        '''Determine whether or not we need to redistribute the ready state'''
        # Try to pre-empty starvation by comparing current RDY against
        # the last value sent.
        alive = [c for c in self.connections() if c.alive()]
        if any(c.ready <= (c.last_ready_sent * 0.25) for c in alive):
            return True