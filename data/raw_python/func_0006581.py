def insert(table, datas, avoid_conflict=False):
        """ Insert row from datas

        :param table: Safe table name
        :param datas: List of dicts.
        :param avoid_conflict: Allows ignoring error if already exists (do nothing then)
        :return:
        """
        if avoid_conflict:
            debut = """INSERT INTO {table} {ENTETE_INSERT} VALUES {BIND_INSERT} ON CONFLICT DO NOTHING"""
        else:
            debut = """INSERT INTO {table} {ENTETE_INSERT} VALUES {BIND_INSERT} RETURNING *"""
        l = [abstractRequetesSQL.formate(debut, table=table, INSERT=d, args=d) for d in datas if d]
        return Executant(l)