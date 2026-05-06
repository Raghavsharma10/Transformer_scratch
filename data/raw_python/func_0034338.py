def _check_add_node(self, parent, name):
        """ Returns False if bad to make name a child of parent """
        if name == ns("text", "a"):
            if parent.tag == ns("draw", "text-box"):
                return False

        return True