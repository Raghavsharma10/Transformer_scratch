def get(self, key, default=None):
        u"""
        Возвращает значение с указанным ключем

        Пример вызова:
        value = self.get('system.database.name')

        :param key: Имя параметра
        :param default: Значение, возвращаемое по умолчанию
        :return: mixed
        """
        segments = key.split('.')
        result = reduce(
            lambda dct, k: dct and dct.get(k) or None,
            segments, self.data)

        return result or default