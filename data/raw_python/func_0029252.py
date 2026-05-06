def get_input_list(self):
        """
        Description:

            Get input list
            Returns an ordered list of all available input keys and names

        """
        inputs = [' '] * len(self.command['input'])
        for key in self.command['input']:
            inputs[self.command['input'][key]['order']] = {"key":key, "name":self.command['input'][key]['name']}
        return inputs