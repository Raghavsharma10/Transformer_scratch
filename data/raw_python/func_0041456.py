def conversion_handler(self, name):
        u"""
        Возвращает обработчик конвертации с указанным именем

        :param name: Имя обработчика
        :return: callable
        """
        try:
            handler = self.conversion_table[name]
        except KeyError:
            raise KeyError((
                u'Конвертирующий тип с именем {} отсутствует '
                u'в таблице соответствия!'
            ).format(name))

        return handler