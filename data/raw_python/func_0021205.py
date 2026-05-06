def list_cmd(only_active, only_aliases, verbose):
    """List indices."""
    def _tree_print(d, rec_list=None, verbose=False, indent=2):
        # Note that on every recursion rec_list is copied,
        # which might not be very effective for very deep dictionaries.
        rec_list = rec_list or []
        for idx, key in enumerate(sorted(d)):
            line = (['│' + ' ' * indent
                     if i == 1 else ' ' * (indent+1) for i in rec_list])
            line.append('└──' if len(d)-1 == idx else '├──')
            click.echo(''.join(line), nl=False)
            if isinstance(d[key], dict):
                click.echo(key)
                new_rec_list = rec_list + [0 if len(d)-1 == idx else 1]
                _tree_print(d[key], new_rec_list, verbose)
            else:
                leaf_txt = '{} -> {}'.format(key, d[key]) if verbose else key
                click.echo(leaf_txt)

    aliases = (current_search.active_aliases
               if only_active else current_search.aliases)
    active_aliases = current_search.active_aliases

    if only_aliases:
        click.echo(json.dumps(list((aliases.keys())), indent=4))
    else:
        # Mark active indices for printout
        aliases = {(k + (' *' if k in active_aliases else '')): v
                   for k, v in aliases.items()}
        click.echo(_tree_print(aliases, verbose=verbose))