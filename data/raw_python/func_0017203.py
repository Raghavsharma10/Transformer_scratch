def set_outgroup(self, outgroup):
        """
        Sets a descendant node as the outgroup of a tree.  This function
        can be used to root a tree or even an internal node.

        Parameters:
        -----------
        outgroup: 
            a node instance within the same tree structure that will be 
            used as a basal node.
        """
        outgroup = _translate_nodes(self, outgroup)

        if self == outgroup:
            ##return
            ## why raise an error for this?
            raise TreeError("Cannot set myself as outgroup")

        parent_outgroup = outgroup.up

        # Detects (sub)tree root
        n = outgroup
        while n.up is not self:
            n = n.up

        # If outgroup is a child from root, but with more than one
        # sister nodes, creates a new node to group them

        self.children.remove(n)
        if len(self.children) != 1:
            down_branch_connector = self.__class__()
            down_branch_connector.dist = 0.0
            down_branch_connector.support = n.support
            for ch in self.get_children():
                down_branch_connector.children.append(ch)
                ch.up = down_branch_connector
                self.children.remove(ch)
        else:
            down_branch_connector = self.children[0]

        # Connects down branch to myself or to outgroup
        quien_va_ser_padre = parent_outgroup
        if quien_va_ser_padre is not self:
            # Parent-child swapping
            quien_va_ser_hijo = quien_va_ser_padre.up
            quien_fue_padre = None
            buffered_dist = quien_va_ser_padre.dist
            buffered_support = quien_va_ser_padre.support

            while quien_va_ser_hijo is not self:
                quien_va_ser_padre.children.append(quien_va_ser_hijo)
                quien_va_ser_hijo.children.remove(quien_va_ser_padre)

                buffered_dist2 = quien_va_ser_hijo.dist
                buffered_support2 = quien_va_ser_hijo.support
                quien_va_ser_hijo.dist = buffered_dist
                quien_va_ser_hijo.support = buffered_support
                buffered_dist = buffered_dist2
                buffered_support = buffered_support2

                quien_va_ser_padre.up = quien_fue_padre
                quien_fue_padre = quien_va_ser_padre

                quien_va_ser_padre = quien_va_ser_hijo
                quien_va_ser_hijo = quien_va_ser_padre.up

            quien_va_ser_padre.children.append(down_branch_connector)
            down_branch_connector.up = quien_va_ser_padre
            quien_va_ser_padre.up = quien_fue_padre

            down_branch_connector.dist += buffered_dist
            outgroup2 = parent_outgroup
            parent_outgroup.children.remove(outgroup)
            outgroup2.dist = 0

        else:
            outgroup2 = down_branch_connector

        outgroup.up = self
        outgroup2.up = self
        # outgroup is always the first children. Some function my
        # trust on this fact, so do no change this.
        self.children = [outgroup,outgroup2]
        middist = (outgroup2.dist + outgroup.dist)/2
        outgroup.dist = middist
        outgroup2.dist = middist
        outgroup2.support = outgroup.support