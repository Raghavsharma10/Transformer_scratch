def winner(self):
        """Returns either x or o if one of them won, otherwise None"""
        for c in 'xo':
            for comb in [(0,3,6), (1,4,7), (2,5,8), (0,1,2), (3,4,5), (6,7,8), (0,4,8), (2,4,6)]:
                if all(self.spots[p] == c for p in comb):
                    return c
        return None