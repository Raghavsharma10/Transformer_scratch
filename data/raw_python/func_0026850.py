def get_db_session(session=None, obj=None):
    """
    utility function that attempts to return sqlalchemy session that could
    have been created/passed in one of few ways:

    * It first tries to read session attached to instance
      if object argument was passed

    * then it tries to return  session passed as argument

    * finally tries to read pylons-like threadlocal called DBSession

    * if this fails exception is thrown

    :param session:
    :param obj:
    :return:
    """
    # try to read the session from instance
    from ziggurat_foundations import models

    if obj:
        return sa.orm.session.object_session(obj)
    # try passed session
    elif session:
        return session
    # try global pylons-like session then
    elif models.DBSession:
        return models.DBSession
    raise ZigguratSessionException("No Session found")