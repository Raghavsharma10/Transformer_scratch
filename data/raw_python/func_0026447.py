def _get_profile(self, user_account):
        """Retrieves a user's profile"""

        try:
            # TODO: Load active profile, not just any
            user_profile = objectmodels['profile'].find_one(
                {'owner': str(user_account.uuid)})
            self.log("Profile: ", user_profile,
                     user_account.uuid, lvl=debug)
        except Exception as e:
            self.log("No profile due to error: ", e, type(e),
                     lvl=error)
            user_profile = None

        if not user_profile:
            default = {
                'uuid': std_uuid(),
                'owner': user_account.uuid,
                'userdata': {
                    'notes': 'Default profile of ' + user_account.name
                }
            }
            user_profile = objectmodels['profile'](default)
            user_profile.save()

        return user_profile