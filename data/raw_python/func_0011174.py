def tab(topics, complete):
    """Utility sub-command for tabcompletion

    This command is meant to be called by a tab completion
    function and is given a the currently entered topics,
    along with a boolean indicating whether or not the
    last entered argument is complete.

    """

    # Discard `be tab`
    topics = list(topics)[2:]

    # When given an incomplete argument,
    # the argument is *sometimes* returned twice (?)
    # .. note:: Seen in Git Bash on Windows
    # $ be in giant [TAB]
    # -> ['giant']
    # $ be in gi[TAB]
    # -> ['gi', 'gi']
    if len(topics) > 1 and topics[-1] == topics[-2]:
        topics.pop()

    # Suggest projects
    if len(topics) == 0:
        projects = lib.list_projects(root=_extern.cwd())
        sys.stdout.write(" ".join(projects))

    elif len(topics) == 1:
        project = topics[0]
        projects = lib.list_projects(root=_extern.cwd())

        # Complete project
        if not complete:
            projects = [i for i in projects if i.startswith(project)]
            sys.stdout.write(" ".join(projects))
        else:
            # Suggest items from inventory
            inventory = _extern.load_inventory(project)
            inventory = lib.list_inventory(inventory)
            items = [i for i, b in inventory]
            sys.stdout.write(" ".join(items))

    else:
        project, item = topics[:2]

        # Complete inventory item
        if len(topics) == 2 and not complete:
            inventory = _extern.load_inventory(project)
            inventory = lib.list_inventory(inventory)
            items = [i for i, b in inventory]
            items = [i for i in items if i.startswith(item)]
            sys.stdout.write(" ".join(items))

        # Suggest items from template
        else:
            try:
                be = _extern.load_be(project)
                templates = _extern.load_templates(project)
                inventory = _extern.load_inventory(project)

                item = topics[-1]
                items = lib.list_template(root=_extern.cwd(),
                                          topics=topics,
                                          templates=templates,
                                          inventory=inventory,
                                          be=be)
                if not complete:
                    items = lib.list_template(root=_extern.cwd(),
                                              topics=topics[:-1],
                                              templates=templates,
                                              inventory=inventory,
                                              be=be)
                    items = [i for i in items if i.startswith(item)]
                    sys.stdout.write(" ".join(items) + " ")
                else:
                    sys.stdout.write(" ".join(items) + " ")

            except IndexError:
                sys.exit(lib.NORMAL)