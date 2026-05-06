def get_bool(self, key, default=None):
        u"""
        Возвращает значение, приведенное к булеву
        """
        return self.get_converted(
            key, ConversionTypeEnum.BOOL, default=default)