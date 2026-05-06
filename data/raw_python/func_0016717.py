def handle_In(self, node):
        '''in'''
        try:
            elts = node.elts
        except AttributeError:
            raise ParseError('Invalid value type for `in` operator: {0}'.format(node.__class__.__name__),
                             col_offset=node.col_offset)
        return {'$in': list(map(self.field.handle, elts))}