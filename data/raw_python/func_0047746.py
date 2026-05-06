def _calc_odds(self):
        '''Calculates the absolute probability of all posible rolls.'''
        def recur(val, h, dice, combinations):
            for pip in dice[0]:
                tot = val + pip
                if len(dice) > 1:
                    combinations = recur(tot, h, dice[1:], combinations)
                else:
                    combinations += 1
                    h[tot] = h.get(tot, 0) + 1
            return combinations
        if self.summable:
            start = 0
        else:
            start = ''
        h = dict()
        funky = [d.values for d in self._dice]
        # count of possible results of rolling dice
        combinations = recur(start, h, funky, 0.0)
        self._odds = [(roll, h[roll], h[roll] / combinations) for roll in h.keys()]
        self._odds.sort()