def TableArgsMeta(table_args):
    '''Declarative metaclass automatically adding (merging) __table_args__ to
    mapped classes. Example:

        Meta = TableArgsMeta({
            'mysql_engine': 'InnoDB',
            'mysql_default charset': 'utf8',
        }

        Base = declarative_base(name='Base', metaclass=Meta)

        class MyClass(Base):
            …

    is equivalent to

        Base = declarative_base(name='Base')

        class MyClass(Base):
            __table_args__ = {
                'mysql_engine': 'InnoDB',
                'mysql_default charset': 'utf8',
            }
            …
    '''

    class _TableArgsMeta(declarative.DeclarativeMeta):

        def __init__(cls, name, bases, dict_):
            if (    # Do not extend base class
                    '_decl_class_registry' not in cls.__dict__ and 
                    # Missing __tablename_ or equal to None means single table
                    # inheritance — no table for it (columns go to table of
                    # base class)
                    cls.__dict__.get('__tablename__') and
                    # Abstract class — no table for it (columns go to table[s]
                    # of subclass[es]
                    not cls.__dict__.get('__abstract__', False)):
                ta = getattr(cls, '__table_args__', {})
                if isinstance(ta, dict):
                    ta = dict(table_args, **ta)
                    cls.__table_args__ = ta
                else:
                    assert isinstance(ta, tuple)
                    if ta and isinstance(ta[-1], dict):
                        tad = dict(table_args, **ta[-1])
                        ta = ta[:-1]
                    else:
                        tad = dict(table_args)
                    cls.__table_args__ = ta + (tad,)
            super(_TableArgsMeta, cls).__init__(name, bases, dict_)

    return _TableArgsMeta