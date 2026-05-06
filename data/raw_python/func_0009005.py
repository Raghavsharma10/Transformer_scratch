def namebase(self):
        """ The same as :meth:`name`, but with one file extension stripped off.

        For example,
        ``Path('/home/guido/python.tar.gz').name == 'python.tar.gz'``,
        but
        ``Path('/home/guido/python.tar.gz').namebase == 'python.tar'``.
        """
        base, ext = self.module.splitext(self.name)
        return base