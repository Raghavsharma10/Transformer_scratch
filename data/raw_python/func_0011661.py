def wait(animation='elipses', text='', speed=0.2):
    """
    Decorator for adding wait animation to long running
    functions.

    Args:
        animation (str, tuple): String reference to animation or tuple
            with custom animation.
        speed (float): Number of seconds each cycle of animation.

    Examples:
        >>> @animation.wait('bar')
        >>> def long_running_function():
        >>>     ... 5 seconds later ...
        >>>     return
    """
    def decorator(func):
        func.animation = animation
        func.speed = speed
        func.text = text

        @wraps(func)
        def wrapper(*args, **kwargs):
            animation = func.animation
            text = func.text
            if not isinstance(animation, (list, tuple)) and \
                    not hasattr(animations, animation):
                text = animation if text == '' else text
                animation = 'elipses'
            wait = Wait(animation=animation, text=text, speed=func.speed)
            wait.start()
            try:
                ret = func(*args, **kwargs)
            finally:
                wait.stop()
            sys.stdout.write('\n')
            return ret
        return wrapper
    return decorator