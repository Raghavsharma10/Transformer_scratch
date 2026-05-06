def stalecheck(web3, **kwargs):
    '''
    Use to require that a function will run only of the blockchain is recently updated.

    If the chain is old, raise a StaleBlockchain exception

    Define how stale the chain can be with keyword arguments from datetime.timedelta,
    like stalecheck(web3, days=2)

    Turn off the staleness check at runtime with:
    wrapped_func(..., assertfresh=False)
    '''
    allowable_delay = datetime.timedelta(**kwargs).total_seconds()

    def decorator(func):
        def wrapper(*args, assertfresh=True, **kwargs):
            if assertfresh:
                last_block = web3.eth.getBlock('latest')
                if not isfresh(last_block, allowable_delay):
                    raise StaleBlockchain(last_block, allowable_delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator