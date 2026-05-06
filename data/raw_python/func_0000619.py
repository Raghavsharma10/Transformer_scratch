def handle_user(self, data):
        '''
        Insert user informations in data

        Override it to add extra user attributes.
        '''
        # Default to unauthenticated anonymous user
        data['user'] = {
            'username': '',
            'is_authenticated': False,
            'is_staff': False,
            'is_superuser': False,
            'permissions': tuple(),
        }
        if 'django.contrib.sessions.middleware.SessionMiddleware' in settings.MIDDLEWARE_CLASSES:
            user = self.request.user
            data['user']['is_authenticated'] = user.is_authenticated()
            if hasattr(user, 'username'):
                data['user']['username'] = user.username
            elif hasattr(user, 'get_username'):
                data['user']['username'] = user.get_username()
            if hasattr(user, 'is_staff'):
                data['user']['is_staff'] = user.is_staff
            if hasattr(user, 'is_superuser'):
                data['user']['is_superuser'] = user.is_superuser
            if hasattr(user, 'get_all_permissions'):
                data['user']['permissions'] = tuple(user.get_all_permissions())