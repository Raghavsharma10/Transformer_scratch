def removeFunction(self):
        """Remove function tag
        """
        root = self.etree
        t_execute = root.find('execute')
        try:
            root.remove(t_execute)
            return True
        except (Exception,) as e:
            print(e)

        return False