def relate(cls, propname, *args, **kwargs):
        """Produce a relationship between this mapped table and another
        one. 
        
        This makes usage of SQLAlchemy's :func:`sqlalchemy.orm.relationship`
        construct.
        
        """
        class_mapper(cls)._configure_property(propname, relationship(*args, **kwargs))