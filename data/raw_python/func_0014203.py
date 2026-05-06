def convert_value(self, value, parameter, request):
        '''
        Converts a parameter value in the view function call.

            value:      value from request.dmp.urlparams to convert
                        The value will always be a string, even if empty '' (never None).

            parameter:  an instance of django_mako_plus.ViewParameter that holds this parameter's
                        name, type, position, etc.

            request:    the current request object.

        "converter functions" register with this class using the @parameter_converter
        decorator.  See converters.py for the built-in converters.

        This function goes through the list of registered converter functions,
        selects the most-specific one that matches the parameter.type, and
        calls it to convert the value.

        If the converter function raises a ValueError, it is caught and
        switched to an Http404 to tell the browser that the requested URL
        doesn't resolve to a page.

        Other useful exceptions that converter functions can raise are:

            Any extension of BaseRedirectException (RedirectException,
                InternalRedirectException, JavascriptRedirectException, ...)
            Http404: returns a Django Http404 response
        '''
        try:
            # we don't convert anything without type hints
            if parameter.type is inspect.Parameter.empty:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug('skipping conversion of parameter `%s` because it has no type hint', parameter.name)
                return value

            # find the converter method for this type
            # I'm iterating through the list to find the most specific match first
            # The list is sorted by specificity so subclasses come before their superclasses
            for ci in self.converters:
                if issubclass(parameter.type, ci.convert_type):
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug('converting parameter `%s` using %s', parameter.name, ci.convert_func)
                    return ci.convert_func(value, parameter)

            # if we get here, there wasn't a converter or this type
            raise ImproperlyConfigured(message='No parameter converter exists for type: {}. Do you need to add an @parameter_converter function for the type?'.format(parameter.type))

        except (BaseRedirectException, Http404):
            log.info('Exception raised during conversion of parameter %s (%s): %s', parameter.position, parameter.name, e)
            raise   # allow these to pass through to the router

        except ValueError as e:
            log.info('ValueError raised during conversion of parameter %s (%s): %s', parameter.position, parameter.name, e)
            raise ConverterHttp404(value, parameter, 'A parameter could not be converted - see the logs for more detail') from e

        except Exception as e:
            log.info('Exception raised during conversion of parameter %s (%s): %s', parameter.position, parameter.name, e)
            raise ConverterException(value, parameter, 'A parameter could not be converted - see the logs for more detail') from e