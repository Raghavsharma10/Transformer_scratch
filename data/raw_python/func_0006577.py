def cree_local_DB(scheme):
    """Create emmpt DB according to the given scheme : dict { table : [ (column_name, column_type), .. ]}
    Usefull at installation of application (and for developement)
    """
    conn = LocalConnexion()
    req = ""
    for table, fields in scheme.items():
        req += f"DROP TABLE IF EXISTS {table};"
        req_fields = ", ".join(f'{c_name} {c_type}' for c_name, c_type in fields)
        req += f"""CREATE TABLE {table} (  {req_fields} ) ;"""
    cur = conn.cursor()
    cur.executescript(req)
    conn.connexion.commit()
    conn.connexion.close()
    logging.info("Database created with succes.")