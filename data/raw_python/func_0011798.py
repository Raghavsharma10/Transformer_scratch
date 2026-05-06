def firstname(self, value, case, gender=None):
        u"""
        Склонение имени

        :param value: Значение для склонения
        :param case: Падеж для склонения (значение из класса Case)
        :param gender: Грамматический род
        """
        if not value:
            raise ValueError('Firstname cannot be empty.')

        return self.__inflect(value, case, 'firstname', gender)