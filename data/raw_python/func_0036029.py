def log_function(self, stream_name, properties={},
                   log_function_stack_trace=False,
                   log_exception_stack_trace=False,
                   namespace=None):
    """
    Logs each call to the function as an event in the stream with name
    `stream_name`. If `log_stack_trace` is set, it will log the stack trace
    under the `stack_trace` key. `properties` is an optional mapping fron key
    name to some function which expects the same arguments as the function
    `function` being decorated. The event will be populated with keys in
    `properties` mapped to the return values of the
    `properties[key_name](*args, **kwargs)`.
    Usage:

      @kronos_client.log_function('mystreamname',
                                  properties={'a': lambda x, y: x,
                                              'b': lambda x, y: y})
      def myfunction(a, b):
        <some code here>
    """
    namespace = namespace or self.namespace

    def decorator(function):
      @functools.wraps(function)
      def wrapper(*args, **kwargs):
        event = {}
        start_time = time.time()
        if log_function_stack_trace:
          event['stack_trace'] = traceback.extract_stack()
        try:
          return function(*args, **kwargs)
        except Exception as exception:
          self._log_exception(event, exception,
                              (sys.last_traceback if log_exception_stack_trace
                               else None))
          raise exception
        finally:
          event['duration'] = time.time() - start_time
          for key, value_getter in properties.iteritems():
            event[key] = value_getter(*args, **kwargs)
          self.put({stream_name: [event]}, namespace=namespace)
      return wrapper
    return decorator