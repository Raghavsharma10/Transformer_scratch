def types(self):
        """All the unique types found in user supplied model"""
        res = []
        for column in self.column_definitions:
            tmp = column.get('type', None)
            res.append(ModelCompiler.get_column_type(tmp)) if tmp else False
        res = list(set(res))
        return res