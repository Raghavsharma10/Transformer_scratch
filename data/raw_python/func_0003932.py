def limit_step(self, step):
        """Clip the a step within the maximum allowed range"""
        if self.qmax is None:
            return step
        else:
            return np.clip(step, -self.qmax, self.qmax)