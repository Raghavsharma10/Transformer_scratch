def get_new_edges(self, subject_graph):
        """Get new edges from the subject graph for the graph search algorithm

           The Graph search algorithm extends the matches iteratively by adding
           matching vertices that are one edge further from the starting vertex
           at each iteration.
        """
        result = []
        #print "Match.get_new_edges self.previous_ends1", self.previous_ends1
        for vertex in self.previous_ends1:
            for neighbor in subject_graph.neighbors[vertex]:
                if neighbor not in self.reverse:
                    result.append((vertex, neighbor))
        return result