def write_build_info():
    """
    Get VCS info from pycbc_glue/generate_vcs_info.py and add build information.
    Substitute these into pycbc_glue/git_version.py.in to produce
    pycbc_glue/git_version.py.
    """
    date = branch = tag = author = committer = status = builder_name = build_date = ""
    id = "1.1.0"
    
    try:
        v = gvcsi.generate_git_version_info()
        id, date, branch, tag, author = v.id, v.date, b.branch, v.tag, v.author
        committer, status = v.committer, v.status

        # determine current time and treat it as the build time
        build_date = time.strftime('%Y-%m-%d %H:%M:%S +0000', time.gmtime())

        # determine builder
        retcode, builder_name = gvcsi.call_out(('git', 'config', 'user.name'))
        if retcode:
            builder_name = "Unknown User"
        retcode, builder_email = gvcsi.call_out(('git', 'config', 'user.email'))
        if retcode:
            builder_email = ""
        builder = "%s <%s>" % (builder_name, builder_email)
    except:
        pass

    sed_cmd = ('sed',
        '-e', 's/@ID@/%s/' % id,
        '-e', 's/@DATE@/%s/' % date,
        '-e', 's/@BRANCH@/%s/' % branch,
        '-e', 's/@TAG@/%s/' % tag,
        '-e', 's/@AUTHOR@/%s/' % author,
        '-e', 's/@COMMITTER@/%s/' % committer,
        '-e', 's/@STATUS@/%s/' % status,
        '-e', 's/@BUILDER@/%s/' % builder_name,
        '-e', 's/@BUILD_DATE@/%s/' % build_date,
        'misc/git_version.py.in')

    # FIXME: subprocess.check_call becomes available in Python 2.5
    sed_retcode = subprocess.call(sed_cmd,
        stdout=open('pycbc_glue/git_version.py', 'w'))
    if sed_retcode:
        raise gvcsi.GitInvocationError
    return id