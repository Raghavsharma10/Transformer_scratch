def ls(*topic, **kwargs):
    """List topic from external datastore

    Arguments:
        topic (str): One or more topics, e.g. ("project", "item", "task")
        root (str, optional): Absolute path to where projects reside,
            defaults to os.getcwd()
        backend (callable, optional): Function to call with absolute path as
            argument to retrieve children. Defaults to os.listdir
        absolute (bool, optional): Whether to return relative or absolute paths

    Example:
        >> ls()
        /projects/thedeal
        /projects/hulk
        >> ls("thedeal")
        /projects/thedeal/assets/ben
        /projects/thedeal/assets/table
        >> ls("thedeal", "ben")
        /projects/thedeal/assets/ben/rigging
        /projects/thedeal/assets/ben/modeling

    """

    context = dump()

    root = kwargs.get("root") or context.get("cwd") or os.getcwd()
    backend = kwargs.get("backend", os.listdir)
    absolute = kwargs.get("absolute", True)

    content = {
        0: "projects",
        1: "inventory",
        2: "template"
    }[min(2, len(topic))]

    # List projects
    if content == "projects":
        projects = lib.list_projects(root=root, backend=backend)
        if absolute:
            return map(lambda p: os.path.join(root, p), projects)
        else:
            return projects

    # List items
    if content == "inventory":
        project = topic[0]

        be = _extern.load(project, "be", root=root)
        inventory = _extern.load(project, "inventory", root=root)
        inventory = lib.invert_inventory(inventory)
        templates = _extern.load(project, "templates", root=root)

        if absolute:
            paths = list()
            for item, binding in inventory.iteritems():
                template = templates.get(binding)
                index = len(topic)
                sliced = lib.slice(index, template)
                paths.append(sliced.format(*(topic + (item,)), **context))
            return paths
        else:
            return inventory.keys()

    # List template
    if content == "template":
        project = topic[0]
        be = _extern.load(project, "be", root=root)
        templates = _extern.load(project, "templates", root=root)
        inventory = _extern.load(project, "inventory", root=root)
        return lib.list_template(root=root,
                                 topics=topic,
                                 templates=templates,
                                 inventory=inventory,
                                 be=be,
                                 absolute=absolute)