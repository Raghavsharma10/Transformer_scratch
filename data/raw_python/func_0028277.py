def create_node(ctx, path):
    """
    Create node for given relative path.

    :param ctx: BuildContext object.

    :param path: Relative path relative to top directory.

    :return: Created Node.
    """
    # Ensure given context object is BuildContext object
    _ensure_build_context(ctx)

    # Get top directory's relative path relative to `wscript` directory
    top_dir_relpath = os.path.relpath(
        # Top directory's absolute path
        ctx.top_dir,
        # `wscript` directory's absolute path
        ctx.run_dir,
    )

    # Convert given relative path to be relative to `wscript` directory
    node_path = os.path.join(top_dir_relpath, path)

    # Create node using the relative path relative to `wscript` directory
    node = ctx.path.make_node(node_path)

    # Return the created node
    return node