def get_jac(self):
        """ Return the jacobian of the expressions """
        if self._jac is True:
            if self.band is None:
                f = self.be.Matrix(self.nf, 1, self.exprs)
                _x = self.be.Matrix(self.nx, 1, self.x)
                return f.jacobian(_x)
            else:
                # Banded
                return self.be.Matrix(banded_jacobian(
                    self.exprs, self.x, *self.band))
        elif self._jac is False:
            return False
        else:
            return self._jac