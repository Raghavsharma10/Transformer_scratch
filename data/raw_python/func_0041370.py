def inline(self, tx_version: int) -> str:
        """
        Return an inline string format of the document

        :param tx_version: Version number of the document
        :return:
        """
        if tx_version == 2:
            return "{0}:{1}:{2}".format(self.source,
                                        self.origin_id,
                                        self.index)
        else:
            return "{0}:{1}:{2}:{3}:{4}".format(self.amount,
                                                self.base,
                                                self.source,
                                                self.origin_id,
                                                self.index)