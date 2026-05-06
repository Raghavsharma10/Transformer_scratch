def find(expression, schema=None):
    '''
    Gets an <expression> and optional <schema>.
    <expression> should be a string of python code.
    <schema> should be a dictionary mapping field names to types.
    '''
    parser = SchemaFreeParser() if schema is None else SchemaAwareParser(schema)
    return parser.parse(expression)