def update(self, gradient, step):
        """Update the search direction given the latest gradient and step"""
        self.old_gradient = self.gradient
        self.gradient = gradient
        N = len(self.gradient)
        if self.inv_hessian is None:
            # update the direction
            self.direction = -self.gradient
            self.status = "SD"
            # new guess of the inverse hessian
            self.inv_hessian = np.identity(N, float)
        else:
            # update the direction
            self.direction = -np.dot(self.inv_hessian, self.gradient)
            self.status = "QN"
            # new guess of the inverse hessian (BFGS)
            y = self.gradient - self.old_gradient
            s = step
            sy = abs(np.dot(s, y))+1e-5
            A = np.outer(-y/sy, s)
            A.ravel()[::N+1] += 1
            self.inv_hessian = (
                np.dot(np.dot(A.transpose(), self.inv_hessian), A) +
                np.outer(s/sy, s)
            )