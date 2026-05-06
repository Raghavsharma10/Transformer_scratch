def use_token(self, token=None):
        """
        Function to use static AUTH_TOKEN as auth for the constructor instead of full login process.

        **Parameters:**:

          - **token**: Static AUTH_TOKEN

        **Returns:** Bool on success or failure. In addition the function will mutate the `cloudgenix.API`
                     constructor items as needed.
        """
        api_logger.info('use_token function:')

        # check token is a string.
        if not isinstance(token, (text_type, binary_type)):
            api_logger.debug('"token" was not a text-style string: {}'.format(text_type(token)))
            return False

        # Start setup of constructor.
        session = self._parent_class.expose_session()

        # clear cookies
        session.cookies.clear()

        # Static Token uses X-Auth-Token header instead of cookies.
        self._parent_class.add_headers({
            'X-Auth-Token': token
        })

        # Step 2: Get operator profile for tenant ID and other info.
        if self.interactive_update_profile_vars():

            # pull tenant detail
            if self._parent_class.tenant_id:

                # add tenant values to API() object
                if self.interactive_tenant_update_vars():

                    # Step 3: Check for ESP/MSP. If so, ask which tenant this session should be for.
                    if self._parent_class.is_esp:
                        # ESP/MSP!
                        choose_status, chosen_client_id = self.interactive_client_choice()

                        if choose_status:
                            # attempt to login as client
                            clogin_resp = self._parent_class.post.login_clients(chosen_client_id, {})

                            if clogin_resp.cgx_status:
                                # login successful, update profile and tenant info
                                c_profile = self.interactive_update_profile_vars()
                                t_profile = self.interactive_tenant_update_vars()

                                if c_profile and t_profile:
                                    # successful full client login.
                                    self._parent_class._password = None
                                    return True

                                else:
                                    if t_profile:
                                        print("ESP Client Tenant detail retrieval failed.")
                                    # clear password out of memory
                                    self._parent_class.email = None
                                    self._parent_class._password = None
                                    return False

                            else:
                                print("ESP Client Login failed.")
                                # clear password out of memory
                                self._parent_class.email = None
                                self._parent_class._password = None
                                return False

                        else:
                            print("ESP Client Choice failed.")
                            # clear password out of memory
                            self._parent_class.email = None
                            self._parent_class._password = None
                            return False

                    # successful!
                    # clear password out of memory
                    self._parent_class._password = None
                    return True

                else:
                    print("Tenant detail retrieval failed.")
                    # clear password out of memory
                    self._parent_class.email = None
                    self._parent_class._password = None
                    return False

        else:
            # Profile detail retrieval failed
            self._parent_class.email = None
            self._parent_class._password = None
            return False

        api_logger.info("EMAIL = %s", self._parent_class.email)
        api_logger.info("USER_ID = %s", self._parent_class._user_id)
        api_logger.info("USER ROLES = %s", json.dumps(self._parent_class.roles))
        api_logger.info("TENANT_ID = %s", self._parent_class.tenant_id)
        api_logger.info("TENANT_NAME = %s", self._parent_class.tenant_name)
        api_logger.info("TOKEN_SESSION = %s", self._parent_class.token_session)

        return True