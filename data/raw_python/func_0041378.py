def inline(self) -> str:
        """
        Return inline string format of the instance

        :return:
        """
        return "{0}:{1}".format(self.index, ' '.join([str(p) for p in self.parameters]))