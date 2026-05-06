def norm(self):
        """Return a Scalar object with the norm of this vector"""
        result = Scalar(self.size, self.deriv)
        result.v = np.sqrt(self.x.v**2 + self.y.v**2 + self.z.v**2)
        if self.deriv > 0:
            result.d += self.x.v*self.x.d
            result.d += self.y.v*self.y.d
            result.d += self.z.v*self.z.d
            result.d /= result.v
        if self.deriv > 1:
            result.dd += self.x.v*self.x.dd
            result.dd += self.y.v*self.y.dd
            result.dd += self.z.v*self.z.dd
            denom = result.v**2
            result.dd += (1 - self.x.v**2/denom)*np.outer(self.x.d, self.x.d)
            result.dd += (1 - self.y.v**2/denom)*np.outer(self.y.d, self.y.d)
            result.dd += (1 - self.z.v**2/denom)*np.outer(self.z.d, self.z.d)
            tmp = -self.x.v*self.y.v/denom*np.outer(self.x.d, self.y.d)
            result.dd += tmp+tmp.transpose()
            tmp = -self.y.v*self.z.v/denom*np.outer(self.y.d, self.z.d)
            result.dd += tmp+tmp.transpose()
            tmp = -self.z.v*self.x.v/denom*np.outer(self.z.d, self.x.d)
            result.dd += tmp+tmp.transpose()
            result.dd /= result.v
        return result