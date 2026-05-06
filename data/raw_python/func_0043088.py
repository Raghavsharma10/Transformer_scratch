def get(self, select):
        """
        :param select: the variable to extract from list
        :return:  a simple list of the extraction
        """
        if is_list(select):
            return [(d[s] for s in select) for d in self.data]
        else:
            return [d[select] for d in self.data]