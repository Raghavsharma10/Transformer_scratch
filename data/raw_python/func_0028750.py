def add_input_nodes(self, input_nodes):
        """
        Set the given nodes as inputs for this node.

        Creates a limited-size queue.Queue for each input node and
        registers each queue as an output of its corresponding node.

        """
        self.input_queues = [queue.Queue(maxsize=3) for _ in input_nodes]
        for input_node, input_queue in zip(input_nodes, self.input_queues):
            input_node.add_output_queue(input_queue)