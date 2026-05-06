def __validate(self, target, value, oldvalue, initiator):
        """ Method executed when the event 'set' is triggered.

        :param target: Object triggered
        :param value: New value
        :param oldvalue: Previous value
        :param initiator: Column modified

        :return: :raise ValidateError:
        """
        if value == oldvalue:
            return value

        if self.allow_null and value is None:
            return value

        if self.check_value(value):
            return value
        else:
            if self.throw_exception:
                if self.message:
                    self.message = self.message.format(
                            field=self.field, new_value=value, old_value=oldvalue, key=initiator.key)
                    raise ValidateError(self.message)
                else:
                    raise ValidateError('Value %s from column %s is not valid' % (value, initiator.key))

            return oldvalue