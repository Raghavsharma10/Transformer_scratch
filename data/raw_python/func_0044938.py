def requires_product_environment(func, *args, **kws):
    """
    task decorator that makes sure that the product environment
    of django_productline is activated:
    -context is bound
    -features have been composed
    """
    from django_productline import startup
    startup.select_product()
    return func(*args, **kws)