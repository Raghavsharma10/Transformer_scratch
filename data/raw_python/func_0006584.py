def supprime(cls,table, **kwargs):
        """ Remove entries matchin given condition
        kwargs is a dict of column name :  value , with length ONE.
        """
        assert len(kwargs) == 1
        field, value = kwargs.popitem()
        req = f"""DELETE FROM {table} WHERE {field} = """ + cls.mark_style
        args = (value,)
        return MonoExecutant((req, args))