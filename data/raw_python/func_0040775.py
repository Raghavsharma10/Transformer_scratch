def run(env):
    """
    Run current project in the Corona Simulator
    """
    os.putenv('CIPR_PACKAGES', env.package_dir)
    os.putenv('CIPR_PROJECT', env.project_directory)

    # `Corona Terminal` doesn't support spaces in filenames so we cd in and use '.'.

    cmd = AND(
        clom.cd(path.dirname(env.project_directory)),
        clom[CORONA_SIMULATOR_PATH](path.basename(env.project_directory))
    )

    try:
        cmd.shell.execute()
    except KeyboardInterrupt:
        pass