def summary(self, compute=False) -> pd.DataFrame:
        """
        :param compute: if should call compute method
        :return:
        """
        if compute or self.result is None:
            self.compute()
        return summary(self.result)