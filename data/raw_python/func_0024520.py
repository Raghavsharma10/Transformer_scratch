def compose(self, data):
        """
        condense reaction container to CGR. see init for details about cgr_type

        :param data: ReactionContainer
        :return: CGRContainer
        """
        g = self.__separate(data) if self.__cgr_type in (1, 2, 3, 4, 5, 6) else self.__condense(data)
        g.meta.update(data.meta)
        return g