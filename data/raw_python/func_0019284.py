def assignrepr(self, prefix):
        """Return a |repr| string with a prefixed assignment."""
        caller = 'Timegrids('
        blanks = ' ' * (len(prefix) + len(caller))
        prefix = f'{prefix}{caller}'
        lines = [f'{self.init.assignrepr(prefix)},']
        if self.sim != self.init:
            lines.append(f'{self.sim.assignrepr(blanks)},')
        lines[-1] = lines[-1][:-1] + ')'
        return '\n'.join(lines)