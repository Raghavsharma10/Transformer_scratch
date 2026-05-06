def emit(only=None, return_vars=None, namespace=None,
         sender=None, post_result_name='result',
         capture_result=True):
    """Decorator to mark a method or function as
    a signal emitter::

        @blinker_herald.emit()
        def create(self, value1, value2):
            return client.do.a.create.post.return.object()

    The above will cause that method to act as::

        # prefixes 'pre' and 'post' + {method name}
        pre_create = _signals.signal('pre_create')
        post_create = _signals.signal('post_create')

        def create(self, value1, value2)
            signals.pre_create.send(self, value1=value1, value2=value2,
              signal_emitter=self)
            result = client.do.a.create.post.return.object()
            signals.post_create.send(
                self, value1=value1, value2=value2, result=result,
                signal_emitter=self
            )
            return result

    If you want to use your own namespace you need to specify in the
    'namespace' argument::
        @signals.emit(namespace=MySignalsNameSpace)

    :param only: can be 'pre' or 'post' and only that signal will emit
    :param return_vars: If not specified all the locals() will be send,
        otherwise only names in the list will be taken from locals()
        can be a dict in form {'pre': ['var1'], 'post': ['var2']}
        if variable is not found it defaults to None
    :param namespace: If not specified namespace will be default
    :param sender: Object or String to be the sender, if not specified
       the first method parameter will be used (commonly self)
    :param post_result_name: the name of the result variable e.g: result
    :param capture_result: Should result be sent to post signals handlers?
    """
    namespace = namespace or signals_namespace
    if not getattr(return_vars, 'get', None):
        return_vars = {'pre': return_vars,
                       'post': return_vars}

    def decorator(fn):
        fnargs = inspect.getargspec(fn).args
        fname = fn.__name__
        fn.pre = namespace.signal('pre_{0}'.format(fname))
        fn.post = namespace.signal('post_{0}'.format(fname))
        if not sender and 'self' not in fnargs and 'cls' not in fnargs:
            raise RuntimeError(
                'functions and static methods requires a sender '
                'e.g: @signals.emit(sender="name or object")'
            )

        def send(action, *a, **kw):
            if only is not None and action != only:
                return
            sig_name = '{0}_{1}'.format(action, fname)
            send_return_vars = return_vars.get(action)
            result = kw.pop(post_result_name, None)
            kw.update(inspect.getcallargs(fn, *a, **kw))
            sendkw = {k: v for k, v in kw.items()
                      if k in (send_return_vars or kw.keys())}
            sendkw['signal_emitter'] = sendkw.pop(
                'self', sendkw.pop('cls', kw.get('self', kw.get('cls', fn))))
            _sender = sender or sendkw['signal_emitter']
            if isinstance(
                _sender, type(SENDER_NAME)
            ) and _sender.__name__ == (SENDER_NAME).__name__:
                _sender = _sender(sendkw['signal_emitter'])
            if capture_result and action == 'post':
                sendkw[post_result_name] = result
            namespace.signal(sig_name).send(_sender, **sendkw)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            send('pre', *args, **kwargs)
            result = fn(*args, **kwargs)
            kwargs[post_result_name] = result
            send('post', *args, **kwargs)
            return result
        return wrapper
    return decorator