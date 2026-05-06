def _register_routes(self, methods):
        """
        _register_routes
        """
        # setup routes by decorator
        methods = [(n, v) for (n, v) in methods if v.__name__ == "wrapper"]
        methods = sorted(methods, key=lambda x: x[1]._order)
        for name, value in methods:
            value()  # execute setting route

        return methods