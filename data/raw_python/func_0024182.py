def atoms_order(self):
        """
        Morgan like algorithm for graph nodes ordering

        :return: dict of atom-weight pairs
        """
        if not len(self):  # for empty containers
            return {}
        elif len(self) == 1:  # optimize single atom containers
            return dict.fromkeys(self, 2)

        params = {n: (int(node), tuple(sorted(int(edge) for edge in self._adj[n].values())))
                  for n, node in self.atoms()}
        newlevels = {}
        countprime = iter(primes)
        weights = {x: newlevels.get(y) or newlevels.setdefault(y, next(countprime))
                   for x, y in sorted(params.items(), key=itemgetter(1))}

        tries = len(self) * 4

        numb = len(set(weights.values()))
        stab = 0

        while tries:
            oldnumb = numb
            neweights = {}
            countprime = iter(primes)

            # weights[n] ** 2 NEED for differentiation of molecules like A-B or any other complete graphs.
            tmp = {n: reduce(mul, (weights[x] for x in m), weights[n] ** 2) for n, m in self._adj.items()}

            weights = {x: (neweights.get(y) or neweights.setdefault(y, next(countprime)))
                       for x, y in sorted(tmp.items(), key=itemgetter(1))}

            numb = len(set(weights.values()))
            if numb == len(self):  # each atom now unique
                break
            elif numb == oldnumb:
                x = Counter(weights.values())
                if x[min(x)] > 1:
                    if stab == 3:
                        break
                elif stab >= 2:
                    break

                stab += 1
            elif stab:
                stab = 0

            tries -= 1
            if not tries and numb < oldnumb:
                warning('morgan. number of attempts exceeded. uniqueness has decreased. next attempt will be made')
                tries = 1
        else:
            warning('morgan. number of attempts exceeded')

        return weights