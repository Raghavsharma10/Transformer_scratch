def remove(path, force=False):
    '''
    Remove file or directory (recursively), if it exists.

    On NFS file systems, if a directory contains :file:`.nfs*` temporary files
    (sometimes created when deleting a file), it waits for them to go away.

    Parameters
    ----------
    path : ~pathlib.Path
        Path to remove.
    force : bool
        If True, will remove files and directories even if they are read-only
        (as if first doing ``chmod -R +w``).
    '''
    if not path.exists():
        return
    else:
        if force:
            with suppress(FileNotFoundError):
                chmod(path, 0o700, '+', recursive=True)
        if path.is_dir() and not path.is_symlink():
            # Note: shutil.rmtree did not handle NFS well

            # First remove all files
            for dir_, dirs, files in os.walk(str(path), topdown=False): # bottom-up walk
                dir_ = Path(dir_)
                for file in files:
                    with suppress(FileNotFoundError):
                        (dir_ / file).unlink()
                for file in dirs:  # Note: os.walk treats symlinks to directories as directories
                    file = dir_ / file
                    if file.is_symlink():
                        with suppress(FileNotFoundError):
                            file.unlink()

            # Now remove all dirs, being careful of any lingering .nfs* files
            for dir_, _, _ in os.walk(str(path), topdown=False): # bottom-up walk
                dir_ = Path(dir_)
                with suppress(FileNotFoundError):
                    # wait for .nfs* files
                    children = list(dir_.iterdir())

                    while children:
                        # only wait for nfs temporary files
                        if any(not child.name.startswith('.nfs') for child in children):
                            dir_.rmdir()  # raises dir not empty

                        # wait and go again
                        time.sleep(.1)
                        children = list(dir_.iterdir())

                    # rm
                    dir_.rmdir()
        else:
            with suppress(FileNotFoundError):
                path.unlink()