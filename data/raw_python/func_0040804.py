def parse_assign(string):
    """Parse an assignment line:

    >>> parse_assign("    scenario8.Actuator_MagazinVacuumOn = TRUE")
    ("scenario8.Actuator_MagazinVacuumOn", "TRUE")
    """
    try:
        a, b = string.split(" = ")
        return a.strip(), b.strip()
    except:
        print("Error with assignment: %s" % string, file=sys.stderr)
        return None, None