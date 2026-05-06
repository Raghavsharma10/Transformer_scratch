def determinize(m):
    """Determinizes a finite automaton."""
    if not m.is_finite():
        raise TypeError("machine must be a finite automaton")

    transitions = collections.defaultdict(lambda: collections.defaultdict(set))
    alphabet = set()
    for transition in m.get_transitions():
        [[lstate], read] = transition.lhs
        [[rstate]] = transition.rhs
        if len(read) > 1:
            raise NotSupportedException("multiple input symbols on transition not supported")
        if len(read) == 1:
            alphabet.add(read[0])
        transitions[lstate][tuple(read)].add(rstate)

    class Set(frozenset):
        def __str__(self):
            return "{{{}}}".format(",".join(map(str, sorted(self))))
        def _repr_html_(self):
            return "{{{}}}".format(",".join(x._repr_html_() for x in sorted(self)))

    def eclosure(states):
        """Find epsilon-closure of set of states"""
        states = set(states)
        frontier = set(states)
        while len(frontier) > 0:
            lstate = frontier.pop()
            for rstate in transitions[lstate][()]:
                if rstate not in states:
                    states.add(rstate)
                    frontier.add(rstate)
        return states

    dm = FiniteAutomaton()

    start_state = Set(eclosure([m.get_start_state()]))
    dm.set_start_state(start_state)

    frontier = {start_state}
    visited = set()
    while len(frontier) > 0:
        lstates = frontier.pop()
        if lstates in visited:
            continue
        visited.add(lstates)
        dtransitions = collections.defaultdict(set)
        for lstate in lstates:
            for read in alphabet:
                dtransitions[read] |= transitions[lstate][(read,)]
        for read in alphabet:
            rstates = Set(eclosure(dtransitions[read]))
            dm.add_transition([[lstates], read], [[rstates]])
            frontier.add(rstates)

    accept_states = set(m.get_accept_states())
    for states in visited:
        if len(states & accept_states) > 0:
            dm.add_accept_state(states)

    return dm