def _init(board, project_dir):
    """
    Initialize an OpenAg-based project.
    Internal function we use for both init and flash commands.
    """
    project_dir = os.path.abspath(project_dir)

    # Initialize the platformio project
    pio_config_path = os.path.join(project_dir, "platformio.ini")
    if not os.path.isfile(pio_config_path):
        click.echo("Initializing PlatformIO project")
        with open("/dev/null", "wb") as null:
            try:
                init = subprocess.Popen(
                    ["platformio", "init", "-b", board], stdin=subprocess.PIPE,
                    stdout=null, cwd=project_dir
                )
                init.communicate("y\n")
            except OSError as e:
                raise RuntimeError("PlatformIO is not installed")
        if init.returncode != 0:
            raise RuntimeError(
                "Failed to initialize PlatformIO project"
            )
    click.echo("OpenAg firmware project initialized!")