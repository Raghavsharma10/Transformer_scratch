def update(cls,table, dic, Id):
        """ Update row with Id from table. Set fields given by dic."""
        if dic:
            req = "UPDATE {table} SET {SET} WHERE id = " + cls.named_style.format('__id') +  " RETURNING * "
            r = abstractRequetesSQL.formate(req, SET=dic, table=table, args=dict(dic, __id=Id))
            return MonoExecutant(r)
        return MonoExecutant((f"SELECT * FROM {table} WHERE id = " + cls.named_style.format('__id'),
                              {"__id": Id}))