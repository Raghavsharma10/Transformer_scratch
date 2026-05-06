def command(state, args):
    """List file priority rules."""
    rules = query.files.get_priority_rules(state.db)
    print(tabulate(rules, headers=['ID', 'Regexp', 'Priority']))