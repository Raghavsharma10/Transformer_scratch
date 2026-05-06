def yum_group_install(**kwargs):
    """ instals a yum group """
    for grp in list(kwargs['groups']):
        log_green("installing %s ..." % grp)
        if 'repo' in kwargs:
            repo = kwargs['repo']
            sudo("yum groupinstall -y --quiet "
                 "--enablerepo=%s '%s'" % (repo, grp))
        else:
            sudo("yum groups mark install -y --quiet '%s'" % grp)
            sudo("yum groups mark convert -y --quiet '%s'" % grp)
            sudo("yum groupinstall -y --quiet '%s'" % grp)