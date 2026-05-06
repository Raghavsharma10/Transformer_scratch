def permissions(self, addr, permissions=None):
        """
        Returns the permissions for a page at address `addr`.

        If optional argument permissions is given, set page permissions to that prior to returning permissions.
        """

        if self.state.solver.symbolic(addr):
            raise SimMemoryError(
                "page permissions cannot currently be looked up for symbolic addresses")

        if isinstance(addr, claripy.ast.bv.BV):
            addr = self.state.solver.eval(addr)

        page_num = addr // self._page_size

        try:
            page = self._get_page(page_num)
        except KeyError:
            raise SimMemoryError("page does not exist at given address")

        # Set permissions for the page
        if permissions is not None:
            if isinstance(permissions, (int, long)):
                permissions = claripy.BVV(permissions, 3)

            if not isinstance(permissions, claripy.ast.bv.BV):
                raise SimMemoryError(
                    "Unknown permissions argument type of {0}.".format(
                        type(permissions)))

            page.permissions = permissions

        return page.permissions