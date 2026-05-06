def equivalent(m1, m2):
    """Hopcroft-Karp algorithm."""
    if not m1.is_finite() and m1.is_deterministic():
        raise TypeError("machine must be a deterministic finite automaton")
    if not m2.is_finite() and m2.is_deterministic():
        raise TypeError("machine must be a deterministic finite automaton")

    # Index transitions. We use tuples (1,q) and (2,q) to rename apart state sets
    alphabet = set()
    d = {}
    for t in m1.get_transitions():
        [[q], a] = t.lhs
        [[r]] = t.rhs
        alphabet.add(a)
        d[(1,q),a] = (1,r)
    for t in m2.get_transitions():
        [[q], a] = t.lhs
        [[r]] = t.rhs
        alphabet.add(a)
        d[(2,q),a] = (2,r)

    # Naive union find data structure
    u = {}
    def union(x, y):
        for z in u:
            if u[z] == x:
                u[z] = y

    for q in m1.states:
        u[1,q] = (1,q)
    for q in m2.states:
        u[2,q] = (2,q)

    s = []

    s1 = (1,m1.get_start_state())
    s2 = (2,m2.get_start_state())
    union(s1, s2)
    s.append((s1, s2))

    while len(s) > 0:
        q1, q2 = s.pop()
        for a in alphabet:
            r1 = u[d[q1,a]]
            r2 = u[d[q2,a]]
            if r1 != r2:
                union(r1, r2)
                s.append((r1, r2))

    cls = {}
    f = ( {(1, q) for q in m1.get_accept_states()} | 
          {(2, q) for q in m2.get_accept_states()} )

    for q in u:
        if u[q] not in cls:
            cls[u[q]] = q in f
        elif (q in f) != cls[u[q]]:
            return False
    return True