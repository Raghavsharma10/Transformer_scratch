def flat_git_tree_to_nested(flat_tree, prefix=''):
    '''
    Given an array in format:
        [
            ["100644", "blob", "ab3ce...", "748", ".gitignore" ],
            ["100644", "blob", "ab3ce...", "748", "path/to/thing" ],
            ...
        ]

    Outputs in a nested format:
        {
            "path": "/",
            "type": "directory",
            "children": [
                {
                    "type": "blob",
                    "size": 748,
                    "sha": "ab3ce...",
                    "mode": "100644",
                },
                ...
            ],
            ...
        }
    '''
    root = _make_empty_dir_dict(prefix if prefix else '/')

    # Filter all descendents of this prefix
    descendent_files = [
        info for info in flat_tree
        if os.path.dirname(info[PATH]).startswith(prefix)
    ]

    # Figure out strictly leaf nodes of this tree (can be immediately added as
    # children)
    children_files = [
        info for info in descendent_files
        if os.path.dirname(info[PATH]) == prefix
    ]

    # Figure out all descendent directories
    descendent_dirs = set(
        os.path.dirname(info[PATH]) for info in descendent_files
        if os.path.dirname(info[PATH]).startswith(prefix)
        and not os.path.dirname(info[PATH]) == prefix
    )

    # Figure out all descendent directories
    children_dirs = set(
        dir_path for dir_path in descendent_dirs
        if os.path.dirname(dir_path) == prefix
    )

    # Recurse into children dirs, constructing file trees for each of them,
    # then appending those
    for dir_path in children_dirs:
        info = flat_git_tree_to_nested(descendent_files, prefix=dir_path)
        root['children'].append(info)

    # Append direct children files
    for info in children_files:
        root['children'].append(_make_child(info))

    return root