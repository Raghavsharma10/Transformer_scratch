def tran3(self, a, b, c, n):
        """Get accumulator for a transition n between chars a, b, c."""
        return (((TRAN[(a+n)&255]^TRAN[b]*(n+n+1))+TRAN[(c)^TRAN[n]])&255)