def execute_sql(
    host,
    username,
    password,
    sql,
    schema=None,
    param=None,
    kwargs=None
):
    """EXECUTE MANY LINES OF SQL (FROM SQLDUMP FILE, MAYBE?"""
    kwargs.schema = coalesce(kwargs.schema, kwargs.database)

    if param:
        with MySQL(kwargs) as temp:
            sql = expand_template(sql, quote_param(param))

    # We have no way to execute an entire SQL file in bulk, so we
    # have to shell out to the commandline client.
    args = [
        "mysql",
        "-h{0}".format(host),
        "-u{0}".format(username),
        "-p{0}".format(password)
    ]
    if schema:
        args.append("{0}".format(schema))

    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=-1
        )
        if is_text(sql):
            sql = sql.encode("utf8")
        (output, _) = proc.communicate(sql)
    except Exception as e:
        raise Log.error("Can not call \"mysql\"", e)

    if proc.returncode:
        if len(sql) > 10000:
            sql = "<" + text_type(len(sql)) + " bytes of sql>"
        Log.error(
            "Unable to execute sql: return code {{return_code}}, {{output}}:\n {{sql}}\n",
            sql=indent(sql),
            return_code=proc.returncode,
            output=output
        )