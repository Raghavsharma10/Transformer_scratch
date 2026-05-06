def git_clean(ctx):
    """
    Delete all files untracked by git.

    :param ctx: Context object.

    :return: None.
    """
    # Get command parts
    cmd_part_s = [
        # Program path
        'git',

        # Clean untracked files
        'clean',

        # Remove all untracked files
        '-x',

        # Remove untracked directories too
        '-d',

        # Force to remove
        '-f',

        # Give two `-f` flags to remove sub-repositories too
        '-f',
    ]

    # Print title
    print_title('git_clean')

    # Print the command in multi-line format
    print_text(_format_multi_line_command(cmd_part_s))

    # Create subprocess to run the command in top directory
    proc = subprocess.Popen(cmd_part_s, cwd=ctx.top_dir)

    # Wait the subprocess to finish
    proc.wait()

    # Print end title
    print_title('git_clean', is_end=True)