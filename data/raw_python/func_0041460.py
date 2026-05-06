def get_int(self, key, default=None):
        u"""
        Возвращает значение, приведенное к числовому
        """
        return self.get_converted(
            key, ConversionTypeEnum.INTEGER, default=default)