def segment(self):
        """An non-overlap version Compror"""

        if not self.seg:
            j = 0
        else:
            j = self.seg[-1][1]
            last_len = self.seg[-1][0]
            if last_len + j > self.n_states:
                return

        i = j
        while j < self.n_states - 1:
            while not (not (i < self.n_states - 1) or not (self.lrs[i + 1] >= i - j + 1)):
                i += 1
            if i == j:
                i += 1
                self.seg.append((0, i))
            else:
                if (self.sfx[i] + self.lrs[i]) <= i:
                    self.seg.append((i - j, self.sfx[i] - i + j + 1))

                else:
                    _i = j + i - self.sfx[i]
                    self.seg.append((_i - j, self.sfx[i] - i + j + 1))
                    _j = _i
                    while not (not (_i < i) or not (self.lrs[_i + 1] - self.lrs[_j] >= _i - _j + 1)):
                        _i += 1
                    if _i == _j:
                        _i += 1
                        self.seg.append((0, _i))
                    else:
                        self.seg.append((_i - _j, self.sfx[_i] - _i + _j + 1))
            j = i
        return self.seg