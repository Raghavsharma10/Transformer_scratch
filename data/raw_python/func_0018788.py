def contains(value: Union[str, 'Type']) -> bool:
        """ Checks if a type is defined """
        if isinstance(value, str):
            return any(value.lower() == i.value for i in Type)

        return any(value == i for i in Type)