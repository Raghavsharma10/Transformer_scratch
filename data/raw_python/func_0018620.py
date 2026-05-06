def add_changes_markup(dom, ins_nodes, del_nodes):
    """
    Add <ins> and <del> tags to the dom to show changes.
    """
    # add markup for inserted and deleted sections
    for node in reversed(del_nodes):
        # diff algorithm deletes nodes in reverse order, so un-reverse the
        # order for this iteration
        insert_or_append(node.orig_parent, node, node.orig_next_sibling)
        wrap(node, 'del')
    for node in ins_nodes:
        wrap(node, 'ins')
    # Perform post-processing and cleanup.
    remove_nesting(dom, 'del')
    remove_nesting(dom, 'ins')
    sort_del_before_ins(dom)
    merge_adjacent(dom, 'del')
    merge_adjacent(dom, 'ins')