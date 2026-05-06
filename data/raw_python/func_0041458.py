def get_converted(self, key, conversion_type, default=None):
        u"""
        Возвращает значение, приведенное к типу,
        соответствующему указанному типу из таблицы соответствия

        :param key: Имя параметра
        :param conversion_type: Имя обработчика конвертации
            из таблицы соответствия
        :param default: Значение по умолчанию
        :return: mixed
        """
        # В случае отсутствия параметра сразу возвращаем значение по умолчанию
        if not self.has_param(key):
            return default

        value = self.get(key, default=default)
        handler = self.conversion_handler(conversion_type)

        try:
            value = handler(value)
        except Exception as exc:
            raise ConversionTypeError((
                u'Произошла ошибка при попытке преобразования типа: {}'
            ).format(exc))

        return value