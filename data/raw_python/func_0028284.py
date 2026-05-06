def virtualenv_setup(
    ctx,
    python,
    inputs=None,
    outputs=None,
    touch=None,
    check_import=False,
    pip_setup_file=None,
    pip_setup_touch=None,
    cache_key=None,
    always=False,
):
    """
    Create task that sets up `virtualenv` package.

    :param ctx: BuildContext object.

    :param python: Python program path.

    :param inputs: Input items list to add to created task.

        See :paramref:`create_cmd_task.inputs` for allowed item types.

    :param outputs: Output items list to add to created task.

        See :paramref:`create_cmd_task.outputs` for allowed item types.

    :param touch: Touch file path for dirty checking.

    :param check_import: Whether import module for dirty checking.

    :param pip_setup_file: `get-pip.py` file path for `pip_setup` task.

    :param pip_setup_touch: Touch file path for `pip_setup` task.

    :param cache_key: Task cache key.

    :param always: Whether always run.

    :return: Created task.
    """
    # Ensure given context object is BuildContext object
    _ensure_build_context(ctx)

    # If `get-pip.py` file path is not given
    if pip_setup_file is None:
        # Not create task that sets up `pip`
        pip_setup_task = None

    # If `get-pip.py` file path is given
    else:
        # Create task that sets up `pip`
        pip_setup_task = pip_setup(
            # Context
            ctx=ctx,

            # Python program path
            python=python,

            # `get-pip.py` file path
            setup_file=pip_setup_file,

            # Touch file path
            touch=pip_setup_touch,

            # Whether import module for dirty checking
            check_import=check_import,

            # Whether always run
            always=always,
        )

    # If touch file path is not given
    if touch is None:
        # Not update touch file
        touch_node = None
    else:
        # Update touch file
        touch_node, always = update_touch_file(
            # Context
            ctx=ctx,

            # Touch file path
            path=touch,

            # Whether import module for dirty checking
            check_import=check_import,

            # Module name to import for dirty checking
            check_import_module='virtualenv',

            # Python program path for dirty checking
            check_import_python=python,

            # Whether always run
            always=always,
        )

    # Create task that sets up `virtualenv` package
    task = create_cmd_task(
        # Context
        ctx=ctx,

        # Command parts
        parts=[
            # Python program path
            python,

            # Run module
            '-m',

            # Module name
            'pip',

            # Install package
            'install',

            # Package name
            'virtualenv',
        ],

        # Input items list
        inputs=[
            # Run after the task that sets up `pip`
            pip_setup_task,

            # Given input items list
            inputs,
        ],

        # Output items list
        outputs=[
            # Use the touch node as output target for dirty checking
            touch_node,

            # Given output items list
            outputs,
        ],

        # Whether always run
        always=always,

        # Task cache key
        cache_key=cache_key or (python, 'virtualenv'),
    )

    # Return the created task
    return task