def get_release_notes(osa_repo_dir, osa_old_commit, osa_new_commit):
    """Get release notes between the two revisions."""
    repo = Repo(osa_repo_dir)

    # Get a list of tags, sorted
    tags = repo.git.tag().split('\n')
    tags = sorted(tags, key=LooseVersion)
    # Currently major tags are being printed after rc and
    # b tags. We need to fix the list so that major
    # tags are printed before rc and b releases
    tags = _fix_tags_list(tags)

    # Find the closest tag from a given SHA
    # The tag found here is the tag that was cut
    # either on or before the given SHA
    checkout(repo, osa_old_commit)
    old_tag = repo.git.describe()

    # If the SHA given is between two release tags, then
    # 'git describe' will return a tag in form of
    # <tag>-<commitNum>-<sha>. For example:
    # 14.0.2-3-g6931e26
    # Since reno does not support this format, we need to
    # strip away the commit number and sha bits.
    if '-' in old_tag:
        old_tag = old_tag[0:old_tag.index('-')]

    # Get the nearest tag associated with the new commit
    checkout(repo, osa_new_commit)
    new_tag = repo.git.describe()
    if '-' in new_tag:
        nearest_new_tag = new_tag[0:new_tag.index('-')]
    else:
        nearest_new_tag = new_tag

    # Truncate the tags list to only include versions
    # between old_sha and new_sha. The latest release
    # is not included in this list. That version will be
    # printed separately in the following step.
    tags = tags[tags.index(old_tag):tags.index(nearest_new_tag)]

    release_notes = ""
    # Checkout the new commit, then run reno to get the latest
    # releasenotes that have been created or updated between
    # the latest release and this new commit.
    repo.git.checkout(osa_new_commit, '-f')
    reno_report_command = ['reno',
                           'report',
                           '--earliest-version',
                           nearest_new_tag]
    reno_report_p = subprocess.Popen(reno_report_command,
                                     cwd=osa_repo_dir,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
    reno_output = reno_report_p.communicate()[0].decode('UTF-8')
    release_notes += reno_output

    # We want to start with the latest packaged release first, so
    # the tags list is reversed
    for version in reversed(tags):
        # If version is an rc or b tag, and it has a major
        # release tag, then skip it. There is no need to print
        # release notes for an rc or b release unless we are
        # comparing shas between two rc or b releases.
        repo.git.checkout(version, '-f')
        # We are outputing one version at a time here
        reno_report_command = ['reno',
                               'report',
                               '--branch',
                               version,
                               '--earliest-version',
                               version]
        reno_report_p = subprocess.Popen(reno_report_command,
                                         cwd=osa_repo_dir,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        reno_output = reno_report_p.communicate()[0].decode('UTF-8')
        # We need to ensure the output includes the version we are concerned
        # about.
        # This is due to https://bugs.launchpad.net/reno/+bug/1670173
        if version in reno_output:
            release_notes += reno_output

    # Clean up "Release Notes" title. We don't need this title for
    # each tagged release.
    release_notes = release_notes.replace(
        "=============\nRelease Notes\n=============",
        ""
    )
    # Replace headers that contain '=' with '~' to comply with osa-differ's
    # formatting
    release_notes = re.sub('===+', _equal_to_tilde, release_notes)
    # Replace headers that contain '-' with '#' to comply with osa-differ's
    # formatting
    release_notes = re.sub('---+', _dash_to_num, release_notes)
    return release_notes