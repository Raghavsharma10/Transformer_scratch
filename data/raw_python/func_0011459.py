def add_type_struct_or_union(self, name, interp, node):
        """Store the node with the name. When it is instantiated,
        the node itself will be handled.

        :name: name of the typedefd struct/union
        :node: the union/struct node
        :interp: the 010 interpreter
        """
        self.add_type_class(name, StructUnionDef(name, interp, node))