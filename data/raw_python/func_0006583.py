def cree(table, dic, avoid_conflict=False):
        """ Create ONE row from dic and returns the entry created """
        if avoid_conflict:
            req = """ INSERT INTO {table} {ENTETE_INSERT} VALUES {BIND_INSERT} ON CONFLICT DO NOTHING RETURNING *"""
        else:
            req = """ INSERT INTO {table} {ENTETE_INSERT} VALUES {BIND_INSERT} RETURNING *"""
        r = abstractRequetesSQL.formate(req, table=table, INSERT=dic, args=dic)
        return MonoExecutant(r)