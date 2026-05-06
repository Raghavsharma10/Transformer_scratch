def dominator_tree(graph):
    """
    RETURN DOMINATOR FOREST
    THERE ARE TWO TREES, "ROOTS" and "LOOPS"
    ROOTS HAVE NO PARENTS
    LOOPS ARE NODES THAT ARE A MEMBER OF A CYCLE THAT HAS NO EXTRNAL PARENT

    roots = dominator_tree(graph).get_children(ROOTS)
    """
    todo = Queue()
    done = set()
    dominator = Tree(None)
    nodes = list(graph.nodes)

    while True:
        # FIGURE OUT NET ITEM TO WORK ON
        if todo:
            node = todo.pop()
        elif nodes:
            node = nodes.pop()
            if len(nodes) % 1000 == 0:
                Log.note("{{num}} nodes remaining", num=len(nodes))
        else:
            break
        if node in done:
            continue

        parents = graph.get_parents(node) - {node}
        if not parents:
            # node WITHOUT parents IS A ROOT
            done.add(node)
            dominator.add_edge(Edge(ROOTS, node))
            continue

        not_done = parents - done
        if not_done:
            # THERE ARE MORE parents TO DO FIRST
            more_todo = not_done - todo
            if not more_todo:
                # ALL PARENTS ARE PART OF A CYCLE, MAKE node A ROOT
                done.add(node)
                dominator.add_edge(Edge(LOOPS, node))
            else:
                # DO THE PARENTS BEFORE node
                todo.push(node)
                for p in more_todo:
                    todo.push(p)
            continue

        # WE CAN GET THE DOMINATORS FOR ALL parents
        if len(parents) == 1:
            # SHORTCUT
            dominator.add_edge(Edge(list(parents)[0], node))
            done.add(node)
            continue

        paths_from_roots = [
            list(reversed(dominator.get_path_to_root(p)))
            for p in parents
        ]

        if any(p[0] is ROOTS for p in paths_from_roots):
            # THIS OBJECT CAN BE REACHED FROM A ROOT, IGNORE PATHS FROM LOOPS
            paths_from_roots = [p for p in paths_from_roots if p[0] is ROOTS]
            if len(paths_from_roots) == 1:
                # SHORTCUT
                dom = paths_from_roots[0][-1]
                dominator.add_edge(Edge(dom, node))
                done.add(node)
                continue

        # FIND COMMON PATH FROM root
        num_paths = len(paths_from_roots)
        for i, x in enumerate(zip_longest(*paths_from_roots)):
            if x.count(x[0]) != num_paths:
                dom = paths_from_roots[0][i-1]
                if dom is LOOPS:
                    # CAN BE REACHED FROM MORE THAN ONE LOOP, PICK ONE TO BLAME
                    dom = paths_from_roots[0][-1]
                break
        else:
            # ALL PATHS IDENTICAL
            dom = paths_from_roots[0][-1]

        dominator.add_edge(Edge(dom, node))
        done.add(node)

    return dominator