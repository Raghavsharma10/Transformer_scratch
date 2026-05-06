def token(cls: Type[OperatorType], keyword: str) -> OperatorType:
        """
        Return Operator instance from keyword

        :param keyword: Operator keyword in expression
        :return:
        """
        op = cls(keyword)
        return op