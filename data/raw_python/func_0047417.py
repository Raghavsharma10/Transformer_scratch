def remove_useless(self):
        """Returns a new grammar containing just useful rules."""
        if not self.is_contextfree():
            raise ValueError("grammar must be context-free")
        by_lhs = collections.defaultdict(list)
        by_rhs = collections.defaultdict(list)
        for [lhs], rhs in self.rules:
            by_lhs[lhs].append((lhs, rhs))
            for y in rhs:
                if y in self.nonterminals:
                    by_rhs[y].append((lhs, rhs))
            
        agenda = collections.deque([self.start])
        reachable = set()
        while len(agenda) > 0:
            x = agenda.popleft()
            if x in reachable: continue
            reachable.add(x)
            for _, rhs in by_lhs[x]:
                for y in rhs:
                    if y in by_lhs:
                        agenda.append(y)

        agenda = collections.deque()
        productive = set()
        for [lhs], rhs in self.rules:
            if all(y not in self.nonterminals for y in rhs):
                agenda.append(lhs)
        while len(agenda) > 0:
            y = agenda.popleft()
            if y in productive: continue
            productive.add(y)
            for lhs, rhs in by_rhs[y]:
                if all(y not in self.nonterminals or y in productive for y in rhs):
                    agenda.append(lhs)

        g = Grammar()
        g.set_start(self.start)

        for [lhs], rhs in self.rules:
            if (lhs in reachable & productive and
                all(y not in self.nonterminals or y in reachable & productive for y in rhs)):
                g.add_rule([lhs], rhs)
        return g