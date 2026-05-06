def active_cosfi(self):
        """
        Takes the average of all instantaneous cosfi values

        Returns
        -------
        float
        """
        inst = self.load_instantaneous()
        values = [float(i['value']) for i in inst if i['key'].endswith('Cosfi')]
        return sum(values) / len(values)