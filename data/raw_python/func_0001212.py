def mkzip(archive, items, mode="w", save_full_paths=False):
    """Recursively zip a directory.

    Args:
        archive (zipfile.ZipFile or str): ZipFile object add to or path to the
            output zip archive.
        items (str or list of str): Single item or list of items (files and
            directories) to be added to zipfile.
        mode (str): w for create new and write a for append to.
        save_full_paths (bool): Preserve full paths.
    """
    close = False
    try:
        if not isinstance(archive, zipfile.ZipFile):
            archive = zipfile.ZipFile(archive, mode, allowZip64=True)
            close = True
        logger.info("mkdzip: Creating %s, from: %s", archive.filename, items)
        if isinstance(items, str):
            items = [items]
        for item in items:
            item = os.path.abspath(item)
            basename = os.path.basename(item)
            if os.path.isdir(item):
                for root, directoires, filenames in os.walk(item):
                    for filename in filenames:
                        path = os.path.join(root, filename)
                        if save_full_paths:
                            archive_path = path.encode("utf-8")
                        else:
                            archive_path = os.path.join(
                                basename, path.replace(item, "").strip("\\/")
                            ).encode("utf-8")
                        archive.write(path, archive_path)
            elif os.path.isfile(item):
                if save_full_paths:
                    archive_name = item.encode("utf-8")
                else:
                    archive_name = basename.encode("utf-8")
                archive.write(item, archive_name)  # , zipfile.ZIP_DEFLATED)
        return True
    except Exception as e:
        logger.error("Error occurred during mkzip: %s" % e)
        return False
    finally:
        if close:
            archive.close()