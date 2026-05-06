def endnodes(self):
        """|Nodes| object containing all |Node| objects currently handled by
        the |HydPy| object which define a downstream end point of a network."""
        endnodes = devicetools.Nodes()
        for node in self.nodes:
            for element in node.exits:
                if ((element in self.elements) and
                        (node not in element.receivers)):
                    break
            else:
                endnodes += node
        return endnodes