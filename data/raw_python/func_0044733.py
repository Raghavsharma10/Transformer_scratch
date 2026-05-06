def _check_node_list(path, sample, template, start_enumerate=0):
    """Check a list of nodes, e.g. function body"""
    if len(sample) != len(template):
        raise ASTNodeListMismatch(path, sample, template)

    for i, (sample_node, template_node) in enumerate(zip(sample, template), start=start_enumerate):
        if callable(template_node):
            # Checker function inside a list
            template_node(sample_node, path+[i])
        else:
            assert_ast_like(sample_node, template_node, path+[i])