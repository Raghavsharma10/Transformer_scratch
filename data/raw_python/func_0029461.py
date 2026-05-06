def _local_map(match, loc: str = 'lr') -> list:
        """
        :param match:
        :param loc: str
            "l" or "r" or "lr"
            turns on/off left/right local area calculation
        :return: list
            list of the same size as the string + 2
            it's the local map that counted { and }
            list can contain: None or int>=0
            from the left of the operator match:
                in `b}a` if a:0 then }:0 and b:1
                in `b{a` if a:0 then {:0 and b:-1(None)
            from the right of the operator match:
                in `a{b` if a:0 then {:0 and b:1
                in `a}b` if a:0 then }:0 and b:-1(None)
            Map for +1 (needed for r'$') and -1 (needed for r'^')
            characters is also stored: +1 -> +1, -1 -> +2
        """
        s = match.string
        map_ = [None] * (len(s) + 2)
        if loc == 'l' or loc == 'lr':
            balance = 0
            for i in reversed(range(0, match.start())):
                map_[i] = balance
                c, prev = s[i], (s[i - 1] if i > 0 else '')
                if (c == '}' or c == '˲') and prev != '\\':
                    balance += 1
                elif (c == '{' or c == '˱') and prev != '\\':
                    balance -= 1
                if balance < 0:
                    break
            map_[-1] = balance
        if loc == 'r' or loc == 'lr':
            balance = 0
            for i in range(match.end(), len(s)):
                map_[i] = balance
                c, prev = s[i], s[i - 1]
                if (c == '{' or c == '˱') and prev != '\\':
                    balance += 1
                elif (c == '}' or c == '˲') and prev != '\\':
                    balance -= 1
                if balance < 0:
                    break
            map_[len(s)] = balance
        return map_