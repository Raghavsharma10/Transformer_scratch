def _line_opt(self):
        """Perform a line search along the current direction"""
        direction = self.search_direction.direction
        if self.constraints is not None:
            try:
                direction = self.constraints.project(self.x, direction)
            except ConstraintError:
                self._screen("CONSTRAINT PROJECT FAILED", newline=True)
                return False
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            return False
        self.line.configure(self.x, direction/direction_norm)

        success, wolfe, qopt, fopt = \
            self.line_search(self.line, self.initial_step_size, self.epsilon)
        if success:
            self.step = qopt*self.line.axis
            self.initial_step_size = np.linalg.norm(self.step)
            self.x = self.x + self.step
            self.f = fopt
            if wolfe:
                self._screen("W")
            else:
                self._screen(" ")
                self.search_direction.reset()
            return True
        else:
            if self.debug_line:
                import matplotlib.pyplot as pt
                import datetime
                pt.clf()
                qs = np.arange(0.0, 100.1)*(5*self.initial_step_size/100.0)
                fs = np.array([self.line(q) for q in qs])
                pt.plot(qs, fs)
                pt.xlim(qs[0], qs[-1])
                fdelta = fs.max() - fs.min()
                if fdelta == 0.0:
                    fdelta = fs.mean()
                fmargin = fdelta*0.1
                pt.ylim(fs.min() - fmargin, fs.max() + fmargin)
                pt.title('fdelta = %.2e   fmean = %.2e' % (fdelta, fs.mean()))
                pt.xlabel('Line coordinate, q')
                pt.ylabel('Function value, f')
                pt.savefig('line_failed_%s.png' % (datetime.datetime.now().isoformat()))
            self._reset_state()
            return False