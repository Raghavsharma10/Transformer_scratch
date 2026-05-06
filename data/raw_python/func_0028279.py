def update_touch_file(
    ctx,
    path,
    check_import=False,
    check_import_module=None,
    check_import_python=None,
    always=False,
):
    """
    Update touch file at given path.

    Do two things:
        - Create touch file if it not exists.
        - Update touch file if import checking fails.

    The returned touch file node is used as task's output target for dirty
    checking. Task will run if the touch file changes.

    :param ctx: BuildContext instance.

    :param path: Touch file relative path relative to top directory.

    :param check_import: Whether import module for dirty checking.

    :param check_import_module: Module name to import for dirty checking.

    :param check_import_python: Python program to use for dirty checking.

    :param always: Whether always run.

    :return: A two-item tuple.

        Tuple format is:
        ::

            (
                touch_file_node,        # Touch file node.
                task_needs_run,         # Whether task needs run.
            )
    """
    # Ensure given context object is BuildContext object
    _ensure_build_context(ctx)

    # Print title
    print_title('Update touch file: {}'.format(path))

    # Create touch node
    touch_node = create_node(ctx, path)

    # Whether task needs run
    need_run = False

    # If the touch file not exists,
    # or `always` flag is on.
    if not touch_node.exists() or always:
        # Set `need_run` flag on
        need_run = True

    # If the touch file exists,
    # and `always` flag is off.
    else:
        # If need import module for dirty checking,
        # and module name to import is given.
        if check_import and check_import_module:
            # Get import statement.
            # Notice `from` import ensures the imported module is not imported
            # as `__main__` module. And `__name__` exists in any module.
            import_stmt = 'from {} import __name__'.format(check_import_module)

            # Print info
            print_text('Check import: {}'.format(import_stmt))

            # If Python program to check import is not given
            if check_import_python is None:
                # Get error message
                msg = (
                    'Error (3BKFW): Python program to check import is not'
                    ' given.'
                )

                # Raise error
                raise ValueError(msg)

            # If Python program to check import is given.

            # Normalize given Python program path
            check_import_python, _ = _normalize_items(
                ctx=ctx,
                items=[check_import_python],
                # Convert node to absolute path
                node_to_str=True,
            )[0]

            # If the Python program path is not string
            if not isinstance(check_import_python, str):
                # Get error message
                msg = (
                    'Error (39FQE): Given Python program to check import is'
                    ' not string or node: {0}.'
                ).format(check_import_python)

                # Raise error
                raise ValueError(msg)

            # If the Python program path is string.

            # If the Python program path is not absolute path
            if not os.path.isabs(check_import_python):
                # Convert the Python program path to absolute path
                check_import_python = \
                    create_node(ctx, check_import_python).abspath()

            # The Python program path is absolute path now.

            # Get command parts
            cmd_part_s = [
                # Python program absolute path
                check_import_python,

                # Run code
                '-c',

                # Code to run
                import_stmt
            ]

            # Print the command in multi-line format
            print_text(_format_multi_line_command(cmd_part_s))

            #
            try:
                # Run the command
                subprocess.check_output(cmd_part_s)

                # If not have error,
                # it means the module can be imported.
                #     Set `need_run` flag off.
                need_run = False

            # If have error,
            # it means the module can not be imported.
            #
            # Notice the program may not exist. So catch general exception.
            except Exception:  # pylint: disable=W0703
                # Set `need_run` flag on
                need_run = True

    # If task needs run
    if need_run:
        # If the touch file's parent directory not exists
        if not touch_node.parent.exists():
            # Create the touch file's parent directory
            touch_node.parent.mkdir()

        # Write current time to the touch file to force content change.
        # This will fail dirty-checking and cause task to run.
        touch_node.write('{0}\n'.format(datetime.utcnow()))

        # Print info
        print_text('Updated.')

    # If task not needs run
    else:
        # Print info
        print_text('Skipped.')

    # Print end title
    print_title('Update touch file: {}'.format(path), is_end=True)

    # Return a two-item tuple
    return touch_node, need_run