def zip_built(outdir):
    """Packages the build folder into a zip"""
    print("Zipping the built files!")

    config_file_dir = os.path.join(cwd, "config.py")
    if not os.path.exists(config_file_dir):
        sys.exit(
            "There dosen't seem to be a configuration file. Have you run the init command?")
    else:
        sys.path.insert(0, cwd)
        try:
            from config import website_name
        except:
            sys.exit(
                "Some of the configuration values could not be found! Maybe your config.py is too old. Run 'blended init' to fix.")

    # Remove the  build folder
    build_dir = os.path.join(cwd, outdir)
    zip_dir = os.path.join(cwd, website_name.replace(" ", "_") + "-build-" +
                           str(datetime.now().date()))
    if os.path.exists(build_dir):
        shutil.make_archive(zip_dir, 'zip', build_dir)
    else:
        print("The " + outdir +
              "/ folder could not be found! Have you run 'blended build' yet?")