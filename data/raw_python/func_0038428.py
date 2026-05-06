def to_dict(self):
        """
        to_dict will clean all protected and private properties
        """
        return dict(
            (k, self.__dict__[k]) for k in self.__dict__ if k.find("_") != 0)