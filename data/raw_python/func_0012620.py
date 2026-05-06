def drawFrom(self, cumsum, r):
        """
        Draws a value from a cumulative sum.
        
        Parameters: 
            cumsum : array
                Cumulative sum from which shall be drawn.

        Returns:
            int : Index of the cumulative sum element drawn.
        """
        a = cumsum.rsplit()
        if len(a)>1:
            b = eval(a[0])[int(a[1])]
        else:
            b = eval(a[0])
            
        return np.nonzero(b>=r)[0][0]