def get_by_type(self, key, conversion_func, default=None):
        u"""
        Возвращает значение, приведенное к типу
        с использованием переданной функции

        :param key: Имя параметра
        :param conversion_func: callable объект,
            принимающий и возвращающий значение
        :param default: Значение по умолчанию
        :return: mixed
        """
        if not self.has_param(key):
            return default

        value = self.get(key, default=default)

        try:
            value = conversion_func(value)
        except Exception as exc:
            raise ConversionTypeError((
                u'Произошла ошибка при попытке преобразования типа: {}'
            ).format(exc))

        return value