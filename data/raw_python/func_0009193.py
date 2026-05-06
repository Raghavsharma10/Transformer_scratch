def optimize(self, x0, target):
        """Calculate an optimum argument of an objective function."""
        x = x0
        for i in range(self.maxiter):
            g = self.g(x, target)
            h = self.h(x, target)
            if i == 0:
                alpha = 0
                m = g
            else:
                alpha = - np.dot(m, np.dot(h, g)) / np.dot(m, np.dot(h, m))
                m = g + np.dot(alpha, m)
            t = - np.dot(m, g) / np.dot(m, np.dot(h, m))
            delta = np.dot(t, m)
            x = x + delta
            if np.linalg.norm(delta) < self.tol:
                break
        return x