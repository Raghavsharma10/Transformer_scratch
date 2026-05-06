def ignore(self, *ignore_lst: str):
        """
        ignore a set of tokens with specific names
        """

        def stream():
            for each in ignore_lst:
                each = ConstStrPool.cast_to_const(each)
                yield id(each), each

        self.ignore_lst.update(stream())