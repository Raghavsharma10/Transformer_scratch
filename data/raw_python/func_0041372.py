def inline(self) -> str:
        """
        Return an inline string format of the document

        :return:
        """
        return "{0}:{1}:{2}".format(self.amount, self.base,
                                    pypeg2.compose(self.condition, output.Condition))