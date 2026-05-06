def kerberos_ccache_init(principal, keytab_file, ccache_file=None):
    """
    Checks whether kerberos credential cache has ticket-granting ticket that is valid for at least
    an hour.

    Default ccache is used unless ccache_file is provided. In that case, KRB5CCNAME environment
    variable is set to the value of ccache_file if we successfully obtain the ticket.
    """
    tgt_valid = False
    env = {"LC_ALL": "C"}  # klist uses locales to format date on RHEL7+
    if ccache_file:
        env["KRB5CCNAME"] = ccache_file

    # check if we have tgt that is valid more than one hour
    rc, klist, _ = run(["klist"], extraenv=env)
    if rc == 0:
        for line in klist.splitlines():
            m = re.match(KLIST_TGT_RE, line)
            if m:
                year = m.group("year")
                if len(year) == 2:
                    year = "20" + year

                expires = datetime.datetime(
                    int(year), int(m.group("month")), int(m.group("day")),
                    int(m.group("hour")), int(m.group("minute")), int(m.group("second"))
                )

                if expires - datetime.datetime.now() > datetime.timedelta(hours=1):
                    logger.debug("Valid TGT found, not renewing")
                    tgt_valid = True
                    break

    if not tgt_valid:
        logger.debug("Retrieving kerberos TGT")
        rc, out, err = run(["kinit", "-k", "-t", keytab_file, principal], extraenv=env)
        if rc != 0:
            raise OsbsException("kinit returned %s:\nstdout: %s\nstderr: %s" % (rc, out, err))

    if ccache_file:
        os.environ["KRB5CCNAME"] = ccache_file