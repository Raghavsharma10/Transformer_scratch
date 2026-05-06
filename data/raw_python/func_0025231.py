def _migrate_library(workspace_dir: pathlib.Path, do_logging: bool=True) -> pathlib.Path:
    """ Migrate library to latest version. """

    library_path_11 = workspace_dir / "Nion Swift Workspace.nslib"
    library_path_12 = workspace_dir / "Nion Swift Library 12.nslib"
    library_path_13 = workspace_dir / "Nion Swift Library 13.nslib"

    library_paths = (library_path_11, library_path_12)
    library_path_latest = library_path_13

    if not os.path.exists(library_path_latest):
        for library_path in reversed(library_paths):
            if os.path.exists(library_path):
                if do_logging:
                    logging.info("Migrating library: %s -> %s", library_path, library_path_latest)
                shutil.copyfile(library_path, library_path_latest)
                break

    return library_path_latest