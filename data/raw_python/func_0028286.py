def pip_ins_req(
    ctx,
    python,
    req_path,
    venv_path=None,
    inputs=None,
    outputs=None,
    touch=None,
    check_import=False,
    check_import_module=None,
    pip_setup_file=None,
    pip_setup_touch=None,
    virtualenv_setup_touch=None,
    always=False,
):
    """
    Create task that uses given virtual environment's `pip` to sets up \
    packages listed in given requirements file.

    :param ctx: BuildContext object.

    :param python: Python program path used to set up `pip` and `virtualenv`.

    :param req_path: Requirements file relative path relative to top directory.

    :param venv_path: Virtual environment directory relative path relative to
        top directory.

        If given, will create the virtual environment and set up packages
        listed in given requirements file in the virtual environment.

        If not given, will set up packages listed in given requirements file in
        given Python program's environment.

    :param inputs: Input items list to add to created task.

        See :paramref:`create_cmd_task.inputs` for allowed item types.

    :param outputs: Output items list to add to created task.

        See :paramref:`create_cmd_task.outputs` for allowed item types.

    :param touch: Touch file path for dirty checking.

    :param check_import: Whether import module for dirty checking.

    :param check_import_module: Module name to import for dirty checking.

    :param pip_setup_file: `get-pip.py` file path for `pip_setup` task.

    :param pip_setup_touch: Touch file path for `pip_setup` task.

    :param virtualenv_setup_touch: Touch file path for `virtualenv_setup` task.

    :param always: Whether always run.

    :return: Created task.
    """
    # Ensure given context object is BuildContext object
    _ensure_build_context(ctx)

    # If virtual environment directory path is not given
    if venv_path is None:
        # Use given Python program path
        venv_python = python

    # If virtual environment directory path is given
    else:
        # Get Python program path in the virtual environment
        venv_python = get_python_path(venv_path)

        # Mark the path as input target
        venv_python = mark_input(venv_python)

    # If virtual environment directory path is not given,
    # it means not create virtual environment.
    if venv_path is None:
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
            always=always,
        )

        # Not create virtual environment
        venv_task = None

    # If virtual environment directory path is given
    else:
        # Not create task that sets up `pip` here because `create_venv`
        # function below will do
        pip_setup_task = None

        # Create task that sets up virtual environment
        venv_task = create_venv(
            # Context
            ctx=ctx,

            # Python program path
            python=python,

            # Virtual environment directory path
            venv_path=venv_path,

            # Output items list
            outputs=[
                # Add the virtual environment's `python` program path as output
                # target for dirty checking
                get_python_path(venv_path),

                # Add the virtual environment's `pip` program path as output
                # target for dirty checking
                get_pip_path(venv_path),
            ],

            # Whether always run
            always=always,

            # Task name
            task_name='Create venv `{}`'.format(venv_path),

            # `get-pip.py` file path for `pip_setup` task
            pip_setup_file=pip_setup_file,

            # Touch file path for `pip_setup` task
            pip_setup_touch=pip_setup_touch,

            # Touch file path for `virtualenv_setup` task
            virtualenv_setup_touch=virtualenv_setup_touch,
        )

    # If touch file path is not given
    if not touch:
        # Not update touch file
        touch_node = None

    # If touch file path is given
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
            check_import_module=check_import_module,

            # Python program path for dirty checking
            check_import_python=venv_python,

            # Whether always run
            always=always,
        )

    # Create task that sets up packages
    task = create_cmd_task(
        # Context
        ctx=ctx,

        # Command parts
        parts=[
            # Python program path
            venv_python,

            # Run module
            '-m',

            # Module name
            'pip',

            # Install package
            'install',

            # Read package names from requirements file
            '-r',

            # Requirements file path. Mark as input target.
            mark_input(req_path),
        ],

        # Input items list
        inputs=inputs,

        # Output items list
        outputs=[
            # Use the touch node as output target for dirty checking
            touch_node,

            # Given output items list
            outputs,
        ],

        # Whether always run
        always=always,
    )

    # Chain these tasks to run one after another
    chain_tasks([
        pip_setup_task,
        venv_task,
        task,
    ])

    # Return the created task
    return task