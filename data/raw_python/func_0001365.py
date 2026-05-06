def run_env_once(f: t.Callable) -> t.Callable:
    """
    A decorator to prevent ``manage.py`` from running code twice for everything.
    (https://stackoverflow.com/questions/16546652/why-does-django-run-everything-twice)

    :param f: function or method to decorate
    :return: callable
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        has_run = os.environ.get(wrapper.__name__)
        if has_run == '1':
            return
        result = f(*args, **kwargs)
        os.environ[wrapper.__name__] = '1'
        return result

    return wrapper