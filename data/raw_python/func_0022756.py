def besj(self, x, n):
        '''
        Function BESJ calculates Bessel function of first kind of order n
        Arguments:
            n - an integer (>=0), the order
            x - value at which the Bessel function is required
        --------------------
        C++ Mathematical Library
        Converted from equivalent FORTRAN library
        Converted by Gareth Walker for use by course 392 computational project
        All functions tested and yield the same results as the corresponding
        FORTRAN versions.

        If you have any problems using these functions please report them to
        M.Muldoon@UMIST.ac.uk

        Documentation available on the web
        http://www.ma.umist.ac.uk/mrm/Teaching/392/libs/392.html
        Version 1.0   8/98
        29 October, 1999
        --------------------
        Adapted for use in AGG library by
                    Andy Wilk (castor.vulgaris@gmail.com)
        Adapted for use in vispy library by
                    Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
        -----------------------------------------------------------------------
        '''
        if n < 0:
            return 0.0

        d = 1e-6
        b = 0
        if math.fabs(x) <= d:
            if n != 0:
                return 0
            return 1

        b1 = 0  # b1 is the value from the previous iteration
        # Set up a starting order for recurrence
        m1 = int(math.fabs(x)) + 6
        if math.fabs(x) > 5:
            m1 = int(math.fabs(1.4 * x + 60 / x))

        m2 = int(n + 2 + math.fabs(x) / 4)
        if m1 > m2:
            m2 = m1

        # Apply recurrence down from curent max order
        while True:
            c3 = 0
            c2 = 1e-30
            c4 = 0
            m8 = 1
            if m2 / 2 * 2 == m2:
                m8 = -1

            imax = m2 - 2
            for i in range(1, imax+1):
                c6 = 2 * (m2 - i) * c2 / x - c3
                c3 = c2
                c2 = c6
                if m2 - i - 1 == n:
                    b = c6
                m8 = -1 * m8
                if m8 > 0:
                    c4 = c4 + 2 * c6

            c6 = 2 * c2 / x - c3
            if n == 0:
                b = c6
            c4 += c6
            b /= c4
            if math.fabs(b - b1) < d:
                return b
            b1 = b
            m2 += 3