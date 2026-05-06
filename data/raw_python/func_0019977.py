def describe(self, name=None):
        """
        Cleanly show what the four displayed distribution moments are:
            - Mean
            - Variance
            - Standardized Skewness Coefficient
            - Standardized Kurtosis Coefficient
        
        For a standard Normal distribution, these are [0, 1, 0, 3].
        
        If the object has an associated tag, this is presented. If the optional
        ``name`` kwarg is utilized, this is presented as with the moments.
        Otherwise, no unique name is presented.
        
        Example
        =======
        ::
        
            >>> x = N(0, 1, 'x')
            >>> x.describe()  # print tag since assigned
            MCERP Uncertain Value (x):
            ...

            >>> x.describe('foobar')  # 'name' kwarg takes precedence
            MCERP Uncertain Value (foobar):
            ...
            
            >>> y = x**2
            >>> y.describe('y')  # print name since assigned
            MCERP Uncertain Value (y):
            ...

            >>> y.describe()  # print nothing since no tag
            MCERP Uncertain Value:
            ...

         """
        mn, vr, sk, kt = self.stats
        if name is not None:
            s = "MCERP Uncertain Value (" + name + "):\n"
        elif self.tag is not None:
            s = "MCERP Uncertain Value (" + self.tag + "):\n"
        else:
            s = "MCERP Uncertain Value:\n"
        s += " > Mean................... {: }\n".format(mn)
        s += " > Variance............... {: }\n".format(vr)
        s += " > Skewness Coefficient... {: }\n".format(sk)
        s += " > Kurtosis Coefficient... {: }\n".format(kt)
        print(s)