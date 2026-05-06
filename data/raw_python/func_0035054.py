def _ir_cum2(self, alpha=1.0):
        code, _ = self.encode()

        N = self.n_states
        BL = np.zeros(N - 1)  # BL is the block length of compror codewords

        h0 = np.log2(np.cumsum(
            [1.0 if sfx == 0 else 0.0 for sfx in self.sfx[1:]])
        )
        """
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.lrs[1:])])
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.max_lrs[1:])])
        h1 = np.array([h if m == 0 else h+np.log2(m) 
                       for h,m in zip(h0,self.avg_lrs[1:])])
        """
        h1 = np.array([np.log2(i + 1) if m == 0 else np.log2(i + 1) + np.log2(m)
                       for i, m in enumerate(self.max_lrs[1:])])

        j = 0
        for i in range(len(code)):
            if self.code[i][0] == 0:
                BL[j] = 1
                j += 1
            else:
                L = code[i][0]
                BL[j:j + L] = L  # range(1,L+1)
                j = j + L

        h1 = h1 / BL
        ir = alpha * h0 - h1
        ir[ir < 0] = 0  # Really a HACK here!!!!!
        return ir, h0, h1