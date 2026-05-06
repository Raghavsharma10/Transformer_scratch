def prin(*args, **kwargs):
    r"""Like ``print``, but a function. I.e. prints out all arguments as
    ``print`` would do. Specify output stream like this::

      print('ERROR', `out="sys.stderr"``).

    """
    print >> kwargs.get('out',None), " ".join([str(arg) for arg in args])