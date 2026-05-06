def register_conversion_handler(self, name, handler):
        u"""
        Регистрация обработчика конвертирования

        :param name: Имя обработчика в таблице соответствия
        :param handler: Обработчик
        """
        if name in self.conversion_table:
            warnings.warn((
                u'Конвертирующий тип с именем {} уже '
                u'существует и будет перезаписан!'
            ).format(name))

        self.conversion_table[name] = handler