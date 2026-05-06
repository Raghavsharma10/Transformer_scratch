def get_details(self):
        """ Overrides the base method in order to give details on the various successes and failures """

        # transform the dictionary of failures into a printable form
        need_to_print_value = True
        failures_for_print = OrderedDict()
        for validator, failure in self.failures.items():
            name = get_callable_name(validator)
            if isinstance(failure, Exception):
                if isinstance(failure, WrappingFailure) or isinstance(failure, CompositionFailure):
                    need_to_print_value = False
                failures_for_print[name] = '{exc_type}: {msg}'.format(exc_type=type(failure).__name__, msg=str(failure))
            else:
                failures_for_print[name] = str(failure)

        if need_to_print_value:
            value_str = ' for value [{val}]'.format(val=self.wrong_value)
        else:
            value_str = ''

        # OrderedDict does not pretty print...
        key_values_str = [repr(key) + ': ' + repr(val) for key, val in failures_for_print.items()]
        failures_for_print_str = '{' + ', '.join(key_values_str) + '}'

        # Note: we do note cite the value in the message since it is most probably available in inner messages [{val}]
        msg = '{what}{possibly_value}. Successes: {success} / Failures: {fails}' \
              ''.format(what=self.get_what(), possibly_value=value_str,
                        success=self.successes, fails=failures_for_print_str)

        return msg