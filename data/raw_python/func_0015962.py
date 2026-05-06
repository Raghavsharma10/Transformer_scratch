def active_power(self):
        """
        Takes the sum of all instantaneous active power values
        Returns them in kWh

        Returns
        -------
        float
        """
        inst = self.load_instantaneous()
        values = [float(i['value']) for i in inst if i['key'].endswith('ActivePower')]
        return sum(values) / 1000