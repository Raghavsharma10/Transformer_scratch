def send_ftp(outdir):
    """Upload the built website to FTP"""
    print("Uploading the files in the " + outdir + "/ directory!\n")

    # Make sure there is actually a configuration file
    config_file_dir = os.path.join(cwd, "config.py")
    if not os.path.exists(config_file_dir):
        sys.exit(
            "There dosen't seem to be a configuration file. Have you run the init command?")
    else:
        sys.path.insert(0, cwd)
        try:
            from config import ftp_server, ftp_username, ftp_password, ftp_port, ftp_upload_path
        except:
            sys.exit(
                "The FTP settings could not be found. Maybe your config file is too old. Re-run 'blended init' to fix it.")

    server = ftp_server
    username = ftp_username
    password = ftp_password
    port = ftp_port

    ftp = FTP()
    ftp.connect(server, port)
    ftp.login(username, password)
    filenameCV = os.path.join(cwd, outdir)

    try:
        ftp.cwd(ftp_upload_path)
        placeFiles(ftp, filenameCV)
    except:
        ftp.quit()
        sys.exit("Files not able to be uploaded! Are you sure the directory exists?")

    ftp.quit()

    print("\nFTP Done!")