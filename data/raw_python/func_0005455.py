def lint_cli(ctx, exclude, skip_untracked, commit_only):
    # type: (click.Context, List[str], bool, bool) -> None
    """ Run pep8 and pylint on all project files.

    You can configure the linting paths using the lint.paths config variable.
    This should be a list of paths that will be linted. If a path to a directory
    is given, all files in that directory and it's subdirectories will be
    used.

    The pep8 and pylint config paths are by default stored in ops/tools/pep8.ini
    and ops/tools/pylint.ini. You can customise those paths in your config with
    lint.pep8_cfg and lint.pylint_cfg variables.

    **Config Example**::

        \b
        lint:
          pylint_cfg: 'ops/tools/pylint.ini'
          pep8_cfg: 'ops/tools/pep8.ini'
          paths:
            - 'src/mypkg'

    **Examples**::

        \b
        $ peltak lint               # Run linter in default mode, skip untracked
        $ peltak lint --commit      # Lint only files staged for commit
        $ peltak lint --all         # Lint all files, including untracked.
        $ peltak lint --pretend     # Print the list of files to lint
        $ peltak lint -e "*.tox*"   # Don't lint files inside .tox directory

    """
    if ctx.invoked_subcommand:
        return

    from peltak.logic import lint
    lint.lint(exclude, skip_untracked, commit_only)