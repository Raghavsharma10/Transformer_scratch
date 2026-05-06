def return_small_clade(treenode):
    "used to produce balanced trees, returns a tip node from the smaller clade"
    node = treenode
    while 1:
        if node.children:
            c1, c2 = node.children
            node = sorted([c1, c2], key=lambda x: len(x.get_leaves()))[0]
        else:
            return node