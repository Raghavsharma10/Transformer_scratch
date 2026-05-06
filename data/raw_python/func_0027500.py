def play_all_validators(self, validators, value):
        """
        Utility method to play all the provided validators on the provided value and output the

        :param validators:
        :param value:
        :return:
        """
        successes = list()
        failures = OrderedDict()
        for validator in validators:
            name = get_callable_name(validator)
            try:
                res = validator(value)
                if result_is_success(res):
                    successes.append(name)
                else:
                    failures[validator] = res

            except Exception as exc:
                failures[validator] = exc

        return successes, failures