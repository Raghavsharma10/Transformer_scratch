def chdir(directory):
    """Change the current working directory.

    Args:
        directory (str): Directory to go to.
    """
    directory = os.path.abspath(directory)
    logger.info("chdir -> %s" % directory)
    try:
        if not os.path.isdir(directory):
            logger.error(
                "chdir -> %s failed! Directory does not exist!", directory
            )
            return False
        os.chdir(directory)
        return True
    except Exception as e:
        logger.error("chdir -> %s failed! %s" % (directory, e))
        return False