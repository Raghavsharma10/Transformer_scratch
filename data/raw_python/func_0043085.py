def get_domain(self, value):
        """
        RETURN domain FOR GIVEN CODOMAIN
        :param value:
        :return:
        """
        return [k for k, v in self.all if v == value]