def info(args):
    " Show information about site. "

    site = find_site(args.PATH)
    print_header("%s -- install information" % site.get_name())
    LOGGER.debug(site.get_info(full=True))
    return True