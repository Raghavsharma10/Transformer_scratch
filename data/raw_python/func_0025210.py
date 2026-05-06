def call(cmd_args, suppress_output=False):
    """ Call an arbitary command and return the exit value, stdout, and stderr as a tuple

    Command can be passed in as either a string or iterable

    >>> result = call('hatchery', suppress_output=True)
    >>> result.exitval
    0
    >>> result = call(['hatchery', 'notreal'])
    >>> result.exitval
    1
    """
    if not funcy.is_list(cmd_args) and not funcy.is_tuple(cmd_args):
        cmd_args = shlex.split(cmd_args)
    logger.info('executing `{}`'.format(' '.join(cmd_args)))
    call_request = CallRequest(cmd_args, suppress_output=suppress_output)
    call_result = call_request.run()
    if call_result.exitval:
        logger.error('`{}` returned error code {}'.format(' '.join(cmd_args), call_result.exitval))
    return call_result