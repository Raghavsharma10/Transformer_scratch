def get_cmd_line(self):
    """
    Return the full command line that will be used when this node
    is run by DAGman.
    """

    cmd = ""
    cmd_list = self.get_cmd_tuple_list()
    for argument in cmd_list:
      cmd += ' '.join(argument) + " "

    return cmd