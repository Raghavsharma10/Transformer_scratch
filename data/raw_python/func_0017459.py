def beta(self):
        '''\
The linear estimation of the parameter vector :math:`\beta` given by

.. math::

    \beta = (X^T X)^-1 X^T y
        
'''
        t = self.X.transpose()
        XX = dot(t,self.X)
        XY = dot(t,self.y)
        return linalg.solve(XX,XY)