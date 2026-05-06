def load_from_stream(self, group):
        """Load a Group from an NCStream object."""
        self._unpack_attrs(group.atts)
        self.name = group.name

        for dim in group.dims:
            new_dim = Dimension(self, dim.name)
            self.dimensions[dim.name] = new_dim
            new_dim.load_from_stream(dim)

        for var in group.vars:
            new_var = Variable(self, var.name)
            self.variables[var.name] = new_var
            new_var.load_from_stream(var)

        for grp in group.groups:
            new_group = Group(self)
            self.groups[grp.name] = new_group
            new_group.load_from_stream(grp)

        for struct in group.structs:
            new_var = Variable(self, struct.name)
            self.variables[struct.name] = new_var
            new_var.load_from_stream(struct)

        if group.enumTypes:
            for en in group.enumTypes:
                self.types[en.name] = enum.Enum(en.name,
                                                [(typ.value, typ.code) for typ in en.map])