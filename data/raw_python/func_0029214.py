def create_apikey_model(user_model):
    """ Generate ApiKey model class and connect it with :user_model:.

    ApiKey is generated with relationship to user model class :user_model:
    as a One-to-One relationship with a backreference.
    ApiKey is set up to be auto-generated when a new :user_model: is created.

    Returns ApiKey document class. If ApiKey is already defined, it is not
    generated.

    Arguments:
        :user_model: Class that represents user model for which api keys will
            be generated and with which ApiKey will have relationship.
    """
    try:
        return engine.get_document_cls('ApiKey')
    except ValueError:
        pass

    fk_kwargs = {
        'ref_column': None,
    }
    if hasattr(user_model, '__tablename__'):
        fk_kwargs['ref_column'] = '.'.join([
            user_model.__tablename__, user_model.pk_field()])
        fk_kwargs['ref_column_type'] = user_model.pk_field_type()

    class ApiKey(engine.BaseDocument):
        __tablename__ = 'nefertari_apikey'

        id = engine.IdField(primary_key=True)
        token = engine.StringField(default=create_apikey_token)
        user = engine.Relationship(
            document=user_model.__name__,
            uselist=False,
            backref_name='api_key',
            backref_uselist=False)
        user_id = engine.ForeignKeyField(
            ref_document=user_model.__name__,
            **fk_kwargs)

        def reset_token(self):
            self.update({'token': create_apikey_token()})
            return self.token

    # Setup ApiKey autogeneration on :user_model: creation
    ApiKey.autogenerate_for(user_model, 'user')

    return ApiKey