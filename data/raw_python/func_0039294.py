def addFunction(self, func_code, from_file=''):
        """Add function 
        """
        if from_file:
            with open(from_file) as f:
                func_code = f.read()

        root = self.etree
        t_execute = root.find('execute')

        if not t_execute :
            t_execute = ctree.SubElement(root, 'execute')

        t_execute.text = "\n\t![CDATA]{0:>5}]]\n\t".format(func_code.ljust(4))

        return True