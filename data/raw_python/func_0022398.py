def tran_hash(self, a, b, c, n):
        """implementation of the tran53 hash function"""
        return (((TRAN[(a+n)&255]^TRAN[b]*(n+n+1))+TRAN[(c)^TRAN[n]])&255)