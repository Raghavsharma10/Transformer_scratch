def fields(self):
        '''A dictionary of fields constructed by this pump'''
        out = dict()
        for operator in self.ops:
            out.update(**operator.fields)

        return out