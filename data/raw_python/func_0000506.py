def get(router, organization, email):
        """
        :rtype: User
        """
        log.info("Getting user: %s" % email)
        resp = router.get_users(org_id=organization.id).json()
        ids = [x['id'] for x in resp if x['email'] == email]
        if len(ids):
            user = User(organization, ids[0]).init_router(router)
            return user
        else:
            raise exceptions.NotFoundError('User with email: %s not found' % email)