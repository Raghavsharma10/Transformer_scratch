def format_function(function):
    '''Returns a cython function from a :attr:`FunctionSpec` instance.
    '''
    args = [format_variable(arg) for arg in function.args]
    return ['{}{} {}({})'.format(
        function.type, function.pointer, function.name, ', '.join(args))]