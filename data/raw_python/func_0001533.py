def connect(self, source, target, witnesses):
        """
        :type source: integer
        :type target: integer
        """
        # print("Adding Edge: "+source+":"+target)
        if self.graph.has_edge(source, target):
            self.graph[source][target]["label"] += ", " + str(witnesses)
        else:
            self.graph.add_edge(source, target, label=witnesses)