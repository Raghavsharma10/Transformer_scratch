def create_venv(
    ctx,
    python,
    venv_path,
    inputs=None,
    outputs=None,
    pip_setup_file=None,
    pip_setup_touch=None,
    virtualenv_setup_touch=None,
    task_name=None,
    cache_key=None,
    always=False,
):
    """
    Create task that sets up virtual environment.

    :param ctx: BuildContext object.

    :param python: Python program path.

    :param venv_path: Virtual environment directory relative path relative to
        top directory.

    :param inputs: Input items list to add to created task.

        See :paramref:`create_cmd_task.inputs` for allowed item types.

    :param outputs: Output items list to add to created task.

        See :paramref:`create_cmd_task.outputs` for allowed item types.

    :param pip_setup_file: `get-pip.py` file path for `pip_setup` task.

    :param pip_setup_touch: Touch file path for `pip_setup` task.

    :param virtualenv_setup_touch: Touch file path for `virtualenv_setup` task.

    :param task_name: Task name for display purpose.

    :param cache_key: Task cache key.

    :param always: Whether always run.

    :return: Created task.
    """
    # Ensure given context object is BuildContext object
    _ensure_build_context(ctx)

    # Create task that sets up `virtualenv` package
    virtualenv_setup_task = virtualenv_setup(
        # Context
        ctx=ctx,

        # Python program path
        python=python,

        # Touch file path
        touch=virtualenv_setup_touch,

        # `get-pip.py` file path for `pip_setup` task.
        pip_setup_file=pip_setup_file,

        # Touch file path for `pip_setup` task.
        pip_setup_touch=pip_setup_touch,
    )

    # Get virtual environment directory path node
    venv_path_node, _ = _normalize_items(
        ctx=ctx,
        items=[venv_path],
        # Convert path string to node
        str_to_node=True
    )[0]

    # Create task that sets up virtual environment.
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
            'virtualenv',

            # Virtual environment directory absolute path
            venv_path_node.abspath(),
        ],

        # Input items list
        inputs=[
            # Run after the task that sets up `virtualenv` package
            virtualenv_setup_task,

            # Given input items list
            inputs,
        ],

        # Output items list
        outputs=[
            # Add the virtual environment's `python` program path as output
            # target for dirty checking
            get_python_path(venv_path),

            # Add the virtual environment's `pip` program path as output target
            # for dirty checking
            get_pip_path(venv_path),

            # Given output items list
            outputs,
        ],

        # Whether always run
        always=always,

        # Task name
        task_name=task_name,

        # Task cache key
        cache_key=cache_key or (python, venv_path),
    )

    # Return the created task
    return task