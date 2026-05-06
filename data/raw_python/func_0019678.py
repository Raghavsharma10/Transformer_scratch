def retrieveVals(self):
        """Retrieve values for graphs."""
        for graph_name in self.getGraphList():
            for field_name in self.getGraphFieldList(graph_name):
                self.setGraphVal(graph_name, field_name, self._stats.get(field_name))