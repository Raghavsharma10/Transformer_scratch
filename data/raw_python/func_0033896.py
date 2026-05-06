def ensure_segment_table(connection):
    """Ensures that the DB represented by connection posses a segment table.
    If not, creates one and prints a warning to stderr"""

    count = connection.cursor().execute("SELECT count(*) FROM sqlite_master WHERE name='segment'").fetchone()[0]

    if count == 0:
        print >>sys.stderr, "WARNING: None of the loaded files contain a segment table"
        theClass  = lsctables.TableByName['segment']
        statement = "CREATE TABLE IF NOT EXISTS segment (" + ", ".join(map(lambda key: "%s %s" % (key, ligolwtypes.ToSQLiteType[theClass.validcolumns[key]]), theClass.validcolumns)) + ")"

        connection.cursor().execute(statement)