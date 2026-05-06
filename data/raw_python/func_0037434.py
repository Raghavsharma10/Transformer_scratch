def is_active(self, key, *instances, **kwargs):
        """
        Returns ``True`` if any of ``instances`` match an active switch.
        Otherwise returns ``False``.

        >>> operator.is_active('my_feature', request) #doctest: +SKIP
        """
        try:
            default = kwargs.pop('default', False)

            # Check all parents for a disabled state
            parts = key.split(':')
            if len(parts) > 1:
                child_kwargs = kwargs.copy()
                child_kwargs['default'] = None
                result = self.is_active(':'.join(parts[:-1]), *instances,
                                        **child_kwargs)

                if result is False:
                    return result
                elif result is True:
                    default = result

            try:
                switch = self[key]
            except KeyError:
                # switch is not defined, defer to parent
                return default

            if switch.status == GLOBAL:
                return True
            elif switch.status == DISABLED:
                return False
            elif switch.status == INHERIT:
                return default

            conditions = switch.value
            # If no conditions are set, we inherit from parents
            if not conditions:
                return default

            instances = list(instances) if instances else []
            instances.extend(self.context.values())

            # check each switch to see if it can execute
            return_value = False

            for namespace, condition in conditions.iteritems():
                condition_set = registry_by_namespace.get(namespace)
                if not condition_set:
                    continue
                result = condition_set.has_active_condition(condition,
                                                            instances)
                if result is False:
                    return False
                elif result is True:
                    return_value = True
        except:
            log.exception('Error checking if switch "%s" is active', key)
            return_value = False

        # there were no matching conditions, so it must not be enabled
        return return_value