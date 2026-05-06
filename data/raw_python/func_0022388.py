def interactive_update_profile_vars(self):
        """
        Function to update the `cloudgenix.API` object with profile info. Run after login or client login.

        **Returns:** Boolean on success/failure,
        """

        profile = self._parent_class.get.profile()

        if profile.cgx_status:

            # if successful, save tenant id and email info to cli state.
            self._parent_class.tenant_id = profile.cgx_content.get('tenant_id')
            self._parent_class.email = profile.cgx_content.get('email')
            self._parent_class._user_id = profile.cgx_content.get('id')
            self._parent_class.roles = profile.cgx_content.get('roles', [])
            self._parent_class.token_session = profile.cgx_content.get('token_session')

            return True

        else:
            print("Profile retrieval failed.")
            # clear password out of memory
            self._parent_class._password = None
            return False