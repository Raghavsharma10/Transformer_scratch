def override_environment(settings, **kwargs):
    # type: (Settings, **str) -> Generator
    """
    Override env vars and reload the Settings object

    NOTE:
    Obviously this context has to be in place before you import any
    module which reads env values at import time.

    NOTE:
    The values in `kwargs` must be strings else you will get a cryptic:

        TypeError: execve() arg 3 contains a non-string value
    """
    old_env = os.environ.copy()
    os.environ.update(kwargs)

    settings._reload()

    try:
        yield
    except Exception:
        raise
    finally:
        for key in kwargs.keys():
            del os.environ[key]
        os.environ.update(old_env)

        settings._reload()