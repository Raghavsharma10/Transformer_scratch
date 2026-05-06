def get_project_urls(project):
    """Get the URLs for all runs from a given project.
    
    TODO: docstring"""
    urls = []
    with ftputil.FTPHost(sra_host, sra_user, sra_password) as ftp_host:
        download_paths = []
        exp_dir = '/sra/sra-instant/reads/ByStudy/sra/SRP/%s/%s/' \
                %(project[:6], project)
        ftp_host.chdir(exp_dir)
        run_folders = ftp_host.listdir(ftp_host.curdir)

        # compile a list of all files
        for folder in run_folders:
            files = ftp_host.listdir(folder)
            assert len(files) == 1
            for f in files:
                path = exp_dir + folder + '/' + f
                urls.append(path)
    return urls