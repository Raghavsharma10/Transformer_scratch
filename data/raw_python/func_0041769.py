def get_arguments(self):
    """Returns the additional options for the grid (such as the queue, memory requirements, ...)."""
    # In python 2, the command line is unicode, which needs to be converted to string before pickling;
    # In python 3, the command line is bytes, which can be pickled directly
    args = loads(self.grid_arguments)['kwargs'] if isinstance(self.grid_arguments, bytes) else loads(self.grid_arguments.encode())['kwargs']
    # in any case, the commands have to be converted to str
    retval = {}
    if 'pe_opt' in args:
      retval['pe_opt'] = args['pe_opt']
    if 'memfree' in args and args['memfree'] is not None:
      retval['memfree'] = args['memfree']
    if 'hvmem' in args and args['hvmem'] is not None:
      retval['hvmem'] = args['hvmem']
    if 'gpumem' in args and args['gpumem'] is not None:
      retval['gpumem'] = args['gpumem']
    if 'env' in args and len(args['env']) > 0:
      retval['env'] = args['env']
    if 'io_big' in args and args['io_big']:
      retval['io_big'] = True

    # also add the queue
    if self.queue_name is not None:
      retval['queue'] = str(self.queue_name)

    return retval