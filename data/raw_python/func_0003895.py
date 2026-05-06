def apply_to(self, x, columns=False):
        """Apply this translation to the given object

           The argument can be several sorts of objects:

           * ``np.array`` with shape (3, )
           * ``np.array`` with shape (N, 3)
           * ``np.array`` with shape (3, N), use ``columns=True``
           * ``Translation``
           * ``Rotation``
           * ``Complete``
           * ``UnitCell``

           In case of arrays, the 3D vectors are translated. In case of trans-
           formations, a new transformation is returned that consists of this
           translation applied AFTER the given translation. In case of a unit
           cell, the original object is returned.

           This method is equivalent to ``self*x``.
        """
        if isinstance(x, np.ndarray) and len(x.shape) == 2 and x.shape[0] == 3 and columns:
            return x + self.t.reshape((3,1))
        if isinstance(x, np.ndarray) and (x.shape == (3, ) or (len(x.shape) == 2 and x.shape[1] == 3)) and not columns:
            return x + self.t
        elif isinstance(x, Complete):
            return Complete(x.r, x.t + self.t)
        elif isinstance(x, Translation):
            return Translation(x.t + self.t)
        elif isinstance(x, Rotation):
            return Complete(x.r, self.t)
        elif isinstance(x, UnitCell):
            return x
        else:
            raise ValueError("Can not apply this translation to %s" % x)