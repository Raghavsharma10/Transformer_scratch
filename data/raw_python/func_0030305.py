def get_scm_info(directory, branch_id=False, read_only=False, filePath=None):
    """
    Reads SCM info from the given directory. It can fill real commit ID into commit_id field or branch name.

    @param directory: directory name
    @param branch_id: reads commit ID if False (default) or branch name if True
    @param read_only: if True it replaces the actual scheme to the read-only for known hosts, e.g. git+ssh to git for
                      git.app.eng.bos.redhat.com, otherwise it just reads it (default)
    @return: an ScmInfo instance
    """
    #TODO use a commit id instead of branch if in detached state

    if (directory, branch_id, read_only, filePath) in scm_info_path_cache:
        return copy.copy(scm_info_path_cache[(directory, branch_id, read_only, filePath)])

    if os.path.exists(os.path.join(directory, ".git")):
        logging.debug("Getting git info for %s", directory)
        if filePath != None:
            args = ["git", "--git-dir", directory + "/.git", "log", "-z", "-n", "2", "--pretty=format:* dummy-branch  %H  %s%n", "--", filePath]
        else:
            args = ["git", "--git-dir", directory + "/.git", "branch", "-v", "--no-abbrev"]

        command = Popen(args, stdout=PIPE, stderr=STDOUT)
        stdout = command.communicate()[0]
        if command.returncode:
            raise ScmException("Reading Git branch name and commit ID from %s failed. Output: %s" % (directory, stdout))

        branch_name = None
        commit_id = None

        for line in stdout.split("\n"):
            if line.startswith("* "):
                pattern = "\* +(.*) +([a-f0-9]{40}) .*"
                m = re.match(pattern, line)
                if m:
                    branch_name = m.group(1).strip()
                    commit_id = m.group(2).strip()
                    break
                else:
                    raise ScmException("Cannot parse commit ID and branch name from result line:\n%s" % line)

        logging.info ("Retrieved branch_name %s and commit_id %s", branch_name, commit_id)

        args = ["git", "--git-dir", directory + "/.git", "remote", "-v"]
        command = Popen(args, stdout=PIPE, stderr=STDOUT)
        stdout = command.communicate()[0]
        if command.returncode:
            raise ScmException("Reading Git remote from %s failed. Output: %s" % (directory, stdout))

        origin_url = None
        for line in stdout.split("\n"):
            if line.startswith("origin" + chr(9)) and line.endswith(" (fetch)"):
                parts = re.split("[\s]+", line, 3)
                origin_url = parts[1]
                break

        if branch_id:
            scminfo = ScmInfo("%s#%s" % (origin_url, branch_name))
        else:
            scminfo = ScmInfo("%s#%s" % (origin_url, commit_id))

        if read_only:
            if scminfo.get_scm_url().startswith("git+ssh://git.app.eng.bos.redhat.com/srv/git/"):
                scminfo.scheme = "git"
                scminfo.path = scminfo.path.replace("/srv/git/", "/")
            elif scminfo.get_scm_url().startswith("git+ssh://code.engineering.redhat.com/"):
                scminfo.scheme = "git+https"
                scminfo.path = ("%s%s" % ("/gerrit/", scminfo.path)).replace("gerrit//", "gerrit/")

        scm_info_path_cache[(directory, branch_id, read_only, filePath)] = scminfo
        return scminfo
    elif os.path.exists(directory):
        #Special case for the integration-platform-tests which test tooling
        #inplace and use the file:// in the test.cfg
        scminfo = ScmInfo("file://%s#%s" % (directory, "xx"))
        scm_info_path_cache[(directory, branch_id, read_only, filePath)] = scminfo
        return scminfo
    else:
        raise ScmException("Unknown SCM type while reading SCM info from %s" % directory)