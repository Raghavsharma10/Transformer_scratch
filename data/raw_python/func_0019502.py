def register_predictor(cls, name):
        """Register method to keep list of predictors."""
        def decorator(subclass):
            """Register as decorator function."""
            cls._predictors[name.lower()] = subclass
            subclass.name = name.lower()
            return subclass
        return decorator