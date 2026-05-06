def token(cls: Type[ConditionType], left: Any, op: Optional[Any] = None,
              right: Optional[Any] = None) -> ConditionType:
        """
        Return Condition instance from arguments and Operator

        :param left: Left argument
        :param op: Operator
        :param right: Right argument
        :return:
        """
        condition = cls()
        condition.left = left
        if op:
            condition.op = op
        if right:
            condition.right = right
        return condition