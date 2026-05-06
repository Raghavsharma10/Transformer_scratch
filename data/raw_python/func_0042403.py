def get_version_from_dirname(name, parent):
    """Extracted sdist"""
    parent = parent.resolve()
    logger.info(f"dirname: Trying to get version of {name} from dirname {parent}")

    name_re = name.replace("_", "[_-]")
    re_dirname = re.compile(f"{name_re}-{RE_VERSION}$")
    if not re_dirname.match(parent.name):
        logger.info(f"dirname: Failed; Does not match {re_dirname!r}")
        return None

    logger.info("dirname: Succeeded")
    return Version.parse(parent.name[len(name) + 1 :])