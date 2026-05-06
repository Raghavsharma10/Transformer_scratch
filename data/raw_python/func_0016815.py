def install_cub(mb_inc_path):
    """ Downloads and installs cub into mb_inc_path """
    cub_url = 'https://github.com/NVlabs/cub/archive/1.6.4.zip'
    cub_sha_hash = '0d5659200132c2576be0b3959383fa756de6105d'
    cub_version_str = 'Current release: v1.6.4 (12/06/2016)'
    cub_zip_file = 'cub.zip'
    cub_zip_dir = 'cub-1.6.4'
    cub_unzipped_path = os.path.join(mb_inc_path, cub_zip_dir)
    cub_new_unzipped_path = os.path.join(mb_inc_path, 'cub')
    cub_header = os.path.join(cub_new_unzipped_path, 'cub', 'cub.cuh')
    cub_readme = os.path.join(cub_new_unzipped_path, 'README.md' )

    # Check for a reasonably valid install
    cub_installed, _ = is_cub_installed(cub_readme, cub_header, cub_version_str)
    if cub_installed:
        log.info("NVIDIA cub installation found "
            "at '{}'".format(cub_new_unzipped_path))
        return

    log.info("No NVIDIA cub installation found")

    # Do we already have a valid cub zip file
    have_valid_cub_file = (os.path.exists(cub_zip_file) and
        os.path.isfile(cub_zip_file) and
        sha_hash_file(cub_zip_file) == cub_sha_hash)

    if have_valid_cub_file:
        log.info("Valid NVIDIA cub archive found '{}'".format(cub_zip_file))
    # Download if we don't have a valid file
    else:
        log.info("Downloading cub archive '{}'".format(cub_url))
        dl_cub(cub_url, cub_zip_file)
        cub_file_sha_hash = sha_hash_file(cub_zip_file)

        # Compare against our supplied hash
        if cub_sha_hash != cub_file_sha_hash:
            msg = ('Hash of file %s downloaded from %s '
                'is %s and does not match the expected '
                'hash of %s. Please manually download '
                'as per the README.md instructions.') % (
                    cub_zip_file, cub_url,
                    cub_file_sha_hash, cub_sha_hash)

            raise InstallCubException(msg)

    # Unzip into montblanc/include/cub
    with zipfile.ZipFile(cub_zip_file, 'r') as zip_file:
        # Remove any existing installs
        shutil.rmtree(cub_unzipped_path, ignore_errors=True)
        shutil.rmtree(cub_new_unzipped_path, ignore_errors=True)

        # Unzip
        zip_file.extractall(mb_inc_path)

        # Rename. cub_unzipped_path is mb_inc_path/cub_zip_dir
        shutil.move(cub_unzipped_path, cub_new_unzipped_path)

        log.info("NVIDIA cub archive unzipped into '{}'".format(
            cub_new_unzipped_path))


    there, reason = is_cub_installed(cub_readme, cub_header, cub_version_str)

    if not there:
        raise InstallCubException(reason)