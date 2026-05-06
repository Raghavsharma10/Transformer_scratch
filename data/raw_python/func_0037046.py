def slack_ver():
    """Open file and read Slackware version
    """
    dist = read_file("/etc/slackware-version")
    sv = re.findall(r"\d+", dist)
    if len(sv) > 2:
        version = (".".join(sv[:2]))
    else:
        version = (".".join(sv))
    return dist.split()[0], version