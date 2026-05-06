def _calc(self, x, y):
        """
        List based implementation of binary tree algorithm for concordance
        measure after :cite:`Christensen2005`.

        """
        x = np.array(x)
        y = np.array(y)
        n = len(y)
        perm = list(range(n))
        perm.sort(key=lambda a: (x[a], y[a]))
        vals = y[perm]
        ExtraY = 0
        ExtraX = 0
        ACount = 0
        BCount = 0
        CCount = 0
        DCount = 0
        ECount = 0
        DCount = 0
        Concordant = 0
        Discordant = 0
        # ids for left child
        li = [None] * (n - 1)
        # ids for right child
        ri = [None] * (n - 1)
        # number of left descendants for a node
        ld = np.zeros(n)
        # number of values equal to value i
        nequal = np.zeros(n)

        for i in range(1, n):
            NumBefore = 0
            NumEqual = 1
            root = 0
            x0 = x[perm[i - 1]]
            y0 = y[perm[i - 1]]
            x1 = x[perm[i]]
            y1 = y[perm[i]]
            if x0 != x1:
                DCount = 0
                ECount = 1
            else:
                if y0 == y1:
                    ECount += 1
                else:
                    DCount += ECount
                    ECount = 1
            root = 0
            inserting = True
            while inserting:
                current = y[perm[i]]
                if current > y[perm[root]]:
                    # right branch
                    NumBefore += 1 + ld[root] + nequal[root]
                    if ri[root] is None:
                        # insert as right child to root
                        ri[root] = i
                        inserting = False
                    else:
                        root = ri[root]
                elif current < y[perm[root]]:
                    # increment number of left descendants
                    ld[root] += 1
                    if li[root] is None:
                        # insert as left child to root
                        li[root] = i
                        inserting = False
                    else:
                        root = li[root]
                elif current == y[perm[root]]:
                    NumBefore += ld[root]
                    NumEqual += nequal[root] + 1
                    nequal[root] += 1
                    inserting = False

            ACount = NumBefore - DCount
            BCount = NumEqual - ECount
            CCount = i - (ACount + BCount + DCount + ECount - 1)
            ExtraY += DCount
            ExtraX += BCount
            Concordant += ACount
            Discordant += CCount

        cd = Concordant + Discordant
        num = Concordant - Discordant
        tau = num / np.sqrt((cd + ExtraX) * (cd + ExtraY))
        v = (4. * n + 10) / (9. * n * (n - 1))
        z = tau / np.sqrt(v)
        pval = erfc(np.abs(z) / 1.4142136)  # follow scipy
        return tau, pval, Concordant, Discordant, ExtraX, ExtraY