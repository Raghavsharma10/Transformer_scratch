def install_provisions(ctx, provision, clear_provisions=False, overwrite=False, list_provisions=False):
    """Install default provisioning data"""

    log("Installing HFOS default provisions")

    # from hfos.logger import verbosity, events
    # verbosity['console'] = verbosity['global'] = events

    from hfos import database
    database.initialize(ctx.obj['dbhost'], ctx.obj['dbname'])

    from hfos.provisions import build_provision_store

    provision_store = build_provision_store()

    def sort_dependencies(items):
        """Topologically sort the dependency tree"""

        g = networkx.DiGraph()
        log('Sorting dependencies')

        for key, item in items:
            log('key: ', key, 'item:', item, pretty=True, lvl=debug)
            dependencies = item.get('dependencies', [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]

            if key not in g:
                g.add_node(key)

            for link in dependencies:
                g.add_edge(key, link)

        if not networkx.is_directed_acyclic_graph(g):
            log('Cycles in provosioning dependency graph detected!', lvl=error)
            log('Involved provisions:', list(networkx.simple_cycles(g)), lvl=error)

        topology = list(networkx.algorithms.topological_sort(g))
        topology.reverse()

        log(topology, pretty=True)

        return topology

    if list_provisions:
        sort_dependencies(provision_store.items())
        exit()

    def provision_item(item):
        """Provision a single provisioning element"""

        method = item.get('method', provisionList)
        model = item.get('model')
        data = item.get('data')

        method(data, model, overwrite=overwrite, clear=clear_provisions)

    if provision is not None:
        if provision in provision_store:
            log("Provisioning ", provision, pretty=True)
            provision_item(provision_store[provision])
        else:
            log("Unknown provision: ", provision, "\nValid provisions are",
                list(provision_store.keys()),
                lvl=error,
                emitter='MANAGE')
    else:
        for name in sort_dependencies(provision_store.items()):
            log("Provisioning", name, pretty=True)
            provision_item(provision_store[name])

    log("Done: Install Provisions")