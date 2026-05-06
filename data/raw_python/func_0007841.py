def isConjNorthNode(self):
        """ Returns if object is conjunct north node. """
        node = self.chart.getObject(const.NORTH_NODE)
        return aspects.hasAspect(self.obj, node, aspList=[0])