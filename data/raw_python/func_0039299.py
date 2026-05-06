def addInput(self, key):
        """Add key to input : key, value or map
        """
        if key not in self.inputs:
            self.inputs.append(key)

        root = self.etree
        t_inputs = root.find('inputs')

        if not t_inputs :
            t_inputs = ctree.SubElement(root, 'inputs')

        t_inputs.append(key.etree)

        return True