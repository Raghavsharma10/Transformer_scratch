def generate_appstruct(self):
        """Returns a Deform compatible appstruct of the object and it's
        properties. Does not recurse into SQLAlchemy relationships.
        An example using the :class:`~drinks.models.User` class (that
        inherits from BaseMixin):

        .. code-block:: python

            >>> user = User(username='mcuserpants', disabled=True)
            >>> user.appstruct
            {'disabled': True, 'username': 'mcuserpants'}

        """
        mapper = object_mapper(self)
        return dict([(p.key, self.__getattribute__(p.key)) for
                     p in mapper.iterate_properties if
                     not self.__getattribute__(p.key) is None])