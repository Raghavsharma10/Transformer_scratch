def isConjSouthNode(self):
        """ Returns if object is conjunct south node. """
        node = self.chart.getObject(const.SOUTH_NODE)
        return aspects.hasAspect(self.obj, node, aspList=[0])