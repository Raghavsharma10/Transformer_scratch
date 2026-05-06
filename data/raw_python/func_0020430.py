def generate_nodeselector_dict(self, nodeselector_str):
        """
        helper method for generating nodeselector dict
        :param nodeselector_str:
        :return: dict
        """
        nodeselector = {}
        if nodeselector_str and nodeselector_str != 'none':
            constraints = [x.strip() for x in nodeselector_str.split(',')]
            raw_nodeselector = dict([constraint.split('=', 1) for constraint in constraints])
            nodeselector = dict([k.strip(), v.strip()] for (k, v) in raw_nodeselector.items())

        return nodeselector