def fetch():
    """Get ChangeLog.txt file size and counts upgraded packages
    """
    mir, r, slackpkg_last_date = mirror(), "", ""
    count, upgraded = 0, []
    if mir:
        tar = urlopen(mir)
        try:
            r = tar.read()
        except AttributeError:
            print("sun: error: can't read mirror")
    if os.path.isfile(var_lib_slackpkg + changelog_txt):
        slackpkg_last_date = read_file("{0}{1}".format(
            var_lib_slackpkg, changelog_txt)).split("\n", 1)[0].strip()
    else:
        return [count, upgraded]
    for line in r.splitlines():
        if slackpkg_last_date == line.strip():
            break
        if (line.endswith("z:  Upgraded.") or line.endswith("z:  Rebuilt.") or
                line.endswith("z:  Added.") or line.endswith("z:  Removed.")):
            upgraded.append(line.split("/")[-1])
            count += 1
        if (line.endswith("*:  Upgraded.") or line.endswith("*:  Rebuilt.") or
                line.endswith("*:  Added.") or line.endswith("*:  Removed.")):
            upgraded.append(line)
            count += 1
    return [count, upgraded]