def backing_type_for(value):
        """Returns the DynamoDB backing type for a given python value's type

        ::

            4 -> 'N'
            ['x', 3] -> 'L'
            {2, 4} -> 'SS'
        """
        if isinstance(value, str):
            vtype = "S"
        elif isinstance(value, bytes):
            vtype = "B"
        # NOTE: numbers.Number check must come **AFTER** bool check since isinstance(True, numbers.Number)
        elif isinstance(value, bool):
            vtype = "BOOL"
        elif isinstance(value, numbers.Number):
            vtype = "N"
        elif isinstance(value, dict):
            vtype = "M"
        elif isinstance(value, list):
            vtype = "L"
        elif isinstance(value, set):
            if not value:
                vtype = "SS"  # doesn't matter, Set(x) should dump an empty set the same for all x
            else:
                inner = next(iter(value))
                if isinstance(inner, str):
                    vtype = "SS"
                elif isinstance(inner, bytes):
                    vtype = "BS"
                elif isinstance(inner, numbers.Number):
                    vtype = "NS"
                else:
                    raise ValueError(f"Unknown set type for inner value {inner!r}")
        else:
            raise ValueError(f"Can't dump unexpected type {type(value)!r} for value {value!r}")
        return vtype