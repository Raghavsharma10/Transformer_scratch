def apply_to(self, x, columns=False):
        """Apply this rotation to the given object

           The argument can be several sorts of objects:

           * ``np.array`` with shape (3, )
           * ``np.array`` with shape (N, 3)
           * ``np.array`` with shape (3, N), use ``columns=True``
           * ``Translation``
           * ``Rotation``
           * ``Complete``
           * ``UnitCell``

           In case of arrays, the 3D vectors are rotated. In case of trans-
           formations, a transformation is returned that consists of this
           rotation applied AFTER the given translation. In case of a unit cell,
           a unit cell with rotated cell vectors is returned.

           This method is equivalent to ``self*x``.
        """
        if isinstance(x, np.ndarray) and len(x.shape) == 2 and x.shape[0] == 3 and columns:
            return np.dot(self.r, x)
        if isinstance(x, np.ndarray) and (x.shape == (3, ) or (len(x.shape) == 2 and x.shape[1] == 3)) and not columns:
            return np.dot(x, self.r.transpose())
        elif isinstance(x, Complete):
            return Complete(np.dot(self.r, x.r), np.dot(self.r, x.t))
        elif isinstance(x, Translation):
            return Complete(self.r, np.dot(self.r, x.t))
        elif isinstance(x, Rotation):
            return Rotation(np.dot(self.r, x.r))
        elif isinstance(x, UnitCell):
            return UnitCell(np.dot(self.r, x.matrix), x.active)
        else:
            raise ValueError("Can not apply this rotation to %s" % x)