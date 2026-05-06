def __remove_args_first_item(self):
        """
        # Todo: finding a better solution
        This is a dirty solution
        Because the first argument of inspectors' args will be itself
        For current implementation, it should be ignore
        """
        if len(self.args) > 0:
            new_args_list = []
            for item in self.args:
                if len(item) > 0 and self.obj == item[0].__class__:
                    new_args_list.append(item[1:])
                else:
                    new_args_list.append(item[:])
            self.__set_args_list(new_args_list)