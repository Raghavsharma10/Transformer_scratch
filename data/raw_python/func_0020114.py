def post_commit(self, results):
        '''\
Process results after a commit.

:parameter results: iterator over :class:`stdnet.instance_session_result`
    items.
:rtype: a two elements tuple containing a list of instances saved and
    a list of ids of instances deleted.'''
        tpy = self._meta.pk_to_python
        instances = []
        deleted = []
        errors = []
        # The length of results must be the same as the length of
        # all committed instances
        for result in results:
            if isinstance(result, Exception):
                errors.append(result.__class__('Exception while committing %s.'
                                               ' %s' % (self._meta, result)))
                continue
            instance = self.pop(result.iid)
            id = tpy(result.id, self.backend)
            if result.deleted:
                deleted.append(id)
            else:
                if instance is None:
                    raise InvalidTransaction('{0} session received id "{1}"\
 which is not in the session.'.format(self, result.iid))
                setattr(instance, instance._meta.pkname(), id)
                instance = self.add(instance,
                                    modified=False,
                                    persistent=result.persistent)
                instance.get_state().score = result.score
                if instance.get_state().persistent:
                    instances.append(instance)
        return instances, deleted, errors