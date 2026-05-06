def create(python, env_dir, system, prompt, bare, virtualenv_py=None):
    """Main entry point to use this as a module.
    """
    if not python or python == sys.executable:
        _create_with_this(
            env_dir=env_dir, system=system, prompt=prompt,
            bare=bare, virtualenv_py=virtualenv_py,
        )
    else:
        _create_with_python(
            python=python,
            env_dir=env_dir, system=system, prompt=prompt,
            bare=bare, virtualenv_py=virtualenv_py,
        )