def routeargs(self, path, routinemethod, container = None, host = None, vhost = None, method = [b'POST'],
                  tostr = True, matchargs = (), fileargs=(), queryargs=(), cookieargs=(), sessionargs=(),
                  csrfcheck = False, csrfarg = '_csrf', formlimit = 67108864):
        '''
        Convenient way to route a processor with arguments. Automatically parse arguments and pass them to
        the corresponding handler arguments. If required arguments are missing, HttpInputException is thrown which
        creates a 400 Bad Request response. If optional arguments are missing, they are replaced with default values
        just as normal Python call does. If handler accepts keyword arguments, extra arguments are sent
        with kwargs. If not, they are safely ignored.
        
        :param path: path to match, can be a regular expression
        
        :param routinemethod: factory function routinemethod(env, arga, argb, argc...). env is an Environment
                object. form or querystring arguments 'arga', 'argb', 'argc' are passed to arga, argb, argc.
        
        :param container: routine container
        
        :param host: if specified, only response to request to specified host
        
        :param vhost: if specified, only response to request to specified vhost.
                      If not specified, response to dispatcher default vhost.
        
        :param method: methods allowed. With POST method, arguments are extracted from form by default;
                        With GET or HEAD method, arguments are extracted from querystring(args).
        
        :param tostr: In Python3, convert bytes to str before sending arguments to handler.
        
        :param matchargs: Instead of using form or args, extract arguments from path match.
                        matchargs is a sequence of matcher group names. If specified a group name
                        by number, the argument is used as positional arguments; if specified a group
                        name by name(str), the argument is used as a keyword argument.
        
        :param fileargs: Instead of using form or args, extract specified arguments from files.
        
        :param queryargs: Instead of using form, extract specified arguments from args. Notice that when
                          GET is allowed, the arguments are always extracted from args by default.
        
        :param cookieargs: Instead of using form or args, extract specified arguments from cookies.
        
        :param sessionargs: Instead of using form or args, extract specified arguments from session.
                            Notice that if sessionargs is not empty, env.sessionstart() is called,
                            so vlcp.service.utils.session.Session module must be loaded.
                
        :param csrfcheck: If True, check <csrfarg> in input arguments against <csrfarg> in session.
                          Notice that csrfcheck=True cause env.sessionstart() to be called, so
                          vlcp.service.utils.session.Session module must be loaded.
        
        :param csrfarg: argument name to check, default to "_csrf"
         
        :param formlimit: limit on parseform, default to 64MB. None to no limit.
        
        For example, if using::
        
           async def handler(env, target, arga, argb, argc):
              ...
           
           dispatcher.routeargs(b'/do/(.*)', handler, matchargs=(1,), queryargs=('argc'))
        
        And there is a HTTP POST::
        
           POST /do/mytarget?argc=1 HTTP/1.1
           Host: ...
           ...
        
           arga=test&argb=test2
        
        then handler accepts arguments: target="mytarget", arga="test", argb="test2", argc="1"
        '''
        code = routinemethod.__code__
        if code.co_flags & 0x08:
            haskwargs = True
        else:
            haskwargs = False
        # Remove argument env
        arguments = code.co_varnames[1:code.co_argcount]
        if hasattr(routinemethod, '__self__') and routinemethod.__self__:
            # First argument is self, remove an extra argument
            arguments=arguments[1:]
        # Optional arguments
        if hasattr(routinemethod, '__defaults__') and routinemethod.__defaults__:
            requires = arguments[:-len(routinemethod.__defaults__)]
        else:
            requires = arguments[:]
        async def handler(env):
            if tostr:
                def _str(s):
                    if not isinstance(s, str):
                        return s.decode(env.encoding)
                    else:
                        return s
            else:
                def _str(s):
                    return s
            if tostr:
                env.argstostr()
                env.cookietostr()
            if env.method == b'POST':
                await env.parseform(formlimit, tostr)
                argfrom = env.form
            else:
                # Ignore input
                env.form = {}
                env.files = {}
                argfrom = env.args
            args = []
            kwargs = dict(argfrom)
            def discard(k):
                if k in kwargs:
                    del kwargs[k]
            def extract(k, source):
                if k in source:
                    kwargs[k] = source[k]
                else:
                    discard(k)
            try:
                ps = 0
                for ma in matchargs:
                    v = _str(env.path_match.group(ma))
                    if v is not None:
                        if isinstance(ma, str):
                            kwargs[ma] = v
                        else:
                            args.append(v)
                            ps += 1
                    else:
                        if isinstance(ma, str):
                            discard(ma)
                        else:
                            if ps < len(arguments):
                                discard(arguments[ps])
                            ps += 1
                for fa in fileargs:
                    extract(fa, env.files)
                if env.method == b'POST':
                    for qa in queryargs:
                        extract(qa, env.args)
                for ca in cookieargs:
                    extract(ca, env.cookies)
                # CSRF check is done before session arguments to prevent check against session self
                if csrfcheck:
                    if csrfarg not in kwargs:
                        raise HttpInputException('CSRF check failed')
                    await env.sessionstart()
                    if env.session.vars[csrfarg] != kwargs[csrfarg]:
                        raise HttpInputException('CSRF check failed')
                if sessionargs:
                    await env.sessionstart()
                    for sa in sessionargs:
                        extract(sa, env.session.vars)
                # Check required arguments
                for k in requires[ps:]:
                    if k not in kwargs:
                        raise HttpInputException('Argument "' + k + '" is required')
                # Remove positional arguments
                for k in requires[:ps]:
                    if k in kwargs:
                        del kwargs[k]
                if not haskwargs:
                    # Remove extra parameters
                    validargs = arguments[ps:]
                    kwargs = dict((k,v) for (k,v) in kwargs.items() if k in validargs)
                r = routinemethod(env, *args, **kwargs)
            except KeyError as exc:
                raise HttpInputException('Missing argument: ' + str(exc))
            except Exception as exc:
                raise HttpInputException(str(exc))
            if r:
                return await r
        self.route(path, handler, container, host, vhost, method)