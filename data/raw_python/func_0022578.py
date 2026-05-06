def normalize(expr):
    """Pass through n-ary expressions, and eliminate empty branches.

    Variadic and binary expressions recursively visit all their children.

    If all children are eliminated then the parent expression is also
    eliminated:

    (& [removed] [removed]) => [removed]

    If only one child is left, it is promoted to replace the parent node:

    (& True) => True
    """
    children = []
    for child in expr.children:
        branch = normalize(child)
        if branch is None:
            continue

        if type(branch) is type(expr):
            children.extend(branch.children)
        else:
            children.append(branch)

    if len(children) == 0:
        return None

    if len(children) == 1:
        return children[0]

    return type(expr)(*children, start=children[0].start,
                      end=children[-1].end)