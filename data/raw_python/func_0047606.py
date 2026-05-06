def run_bfs(m, w, trace=False, steps=1000):
    """Runs an automaton using breadth-first search."""
    from .machines import Store, Configuration, Transition

    agenda = collections.deque()
    chart = {}

    # Initial configuration
    config = list(m.start_config)
    w = Store(w)
    config[m.input] = w
    config = Configuration(config)

    chart[config] = 0
    agenda.append(config)
    run = graphs.Graph()
    run.attrs['rankdir'] = 'LR'
    run.add_node(config, {'start': True})

    while len(agenda) > 0:
        tconfig = agenda.popleft()

        if trace: print("trigger: {}".format(tconfig))

        for aconfig in m.accept_configs:
            if aconfig.match(tconfig):
                run.add_node(tconfig, {'accept': True})

        if chart[tconfig] == steps:
            if trace: print("maximum number of steps reached")
            run.add_node(tconfig, {'incomplete': True})
            continue

        for rule in m.transitions:
            if trace: print("rule: {}".format(rule))
            if rule.match(tconfig):
                nconfig = rule.apply(tconfig)

                if nconfig in chart:
                    assert chart[nconfig] <= chart[tconfig]+1
                    if trace: print("merge: {}".format(nconfig))
                else:
                    chart[nconfig] = chart[tconfig]+1
                    if trace: print("add: {}".format(nconfig))
                    agenda.append(nconfig)
                run.add_edge(tconfig, nconfig)

    # If input tape is one-way, then rank all nodes by input position
    if m.oneway:
        for q in run.nodes:
            ql = list(q)
            run.nodes[q]['rank'] = ql.pop(m.input)
            run.nodes[q]['label'] = Configuration(ql)
        for i in range(len(w)+1):
            r = 'rank{}'.format(i)
            run.add_node(r, {'rank' : Store(w[i:]), 'style' : 'invisible'})
            if i > 0:
                run.add_edge(rprev, r, {'color': 'white', 'label' : w[i-1]})
            rprev = r

    return run