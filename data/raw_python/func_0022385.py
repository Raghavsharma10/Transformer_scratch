def login(self, email=None, password=None):
        """
        Interactive login using the `cloudgenix.API` object. This function is more robust and handles SAML and MSP accounts.
        Expects interactive capability. if this is not available, use `cloudenix.API.post.login` directly.

        **Parameters:**:

          - **email**: Email to log in for, will prompt if not entered.
          - **password**: Password to log in with, will prompt if not entered. Ignored for SAML v2.0 users.

        **Returns:** Bool. In addition the function will mutate the `cloudgenix.API` constructor items as needed.
        """
        # if email not given in function, or if first login fails, prompt.

        if email is None:
            # If user is not set, pull from cache. If not in cache, prompt.
            if self._parent_class.email:
                email = self._parent_class.email
            else:
                email = compat_input("login: ")

        if password is None:
            # if pass not given on function, or if first login fails, prompt.
            if self._parent_class._password:
                password = self._parent_class._password
            else:
                password = getpass.getpass()

        # Try and login
        # For SAML 2.0 support, set the Referer URL prior to logging in.
        # add referer header to the session.
        self._parent_class.add_headers({'Referer': "{}/v2.0/api/login".format(self._parent_class.controller)})

        # call the login API.
        response = self._parent_class.post.login({"email": email, "password": password})

        if response.cgx_status:

            # Check for SAML 2.0 login
            if not response.cgx_content.get('x_auth_token'):
                urlpath = response.cgx_content.get("urlpath", "")
                request_id = response.cgx_content.get("requestId", "")
                if urlpath and request_id:
                    # SAML 2.0
                    print('SAML 2.0: To finish login open the following link in a browser\n\n{0}\n\n'.format(urlpath))
                    found_auth_token = False
                    for i in range(20):
                        print('Waiting for {0} seconds for authentication...'.format((20 - i) * 5))
                        saml_response = self.check_sso_login(email, request_id)
                        if saml_response.cgx_status and saml_response.cgx_content.get('x_auth_token'):
                            found_auth_token = True
                            break
                        # wait before retry.
                        time.sleep(5)
                    if not found_auth_token:
                        print("Login time expired! Please re-login.\n")
                        # log response when debug
                        try:
                            api_logger.debug("LOGIN_FAIL_RESPONSE = %s", json.dumps(response, indent=4))
                        except (TypeError, ValueError):
                            # not JSON response, don't pretty print log.
                            api_logger.debug("LOGIN_FAIL_RESPONSE = %s", str(response))
                        # print login error
                        print('Login failed, please try again', response)
                        # Flush command-line entered login info if failure.
                        self._parent_class.email = None
                        self._parent_class.password = None
                        return False

            api_logger.info('Login successful:')
            # if we got here, we either got an x_auth_token in the original login, or
            # we got an auth_token cookie set via SAML. Figure out which.
            auth_token = response.cgx_content.get('x_auth_token')
            if auth_token:
                # token in the original login (not saml) means region parsing has not been done.
                # do now, and recheck if cookie needs set.
                auth_region = self._parent_class.parse_region(response)
                self._parent_class.update_region_to_controller(auth_region)
                self._parent_class.reparse_login_cookie_after_region_update(response)
            # debug info if needed
            api_logger.debug("AUTH_TOKEN=%s", response.cgx_content.get('x_auth_token'))

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
                                        # remove referer header prior to continuing.
                                        self._parent_class.remove_header('Referer')
                                        return True

                                    else:
                                        if t_profile:
                                            print("ESP Client Tenant detail retrieval failed.")
                                        # clear password out of memory
                                        self._parent_class.email = None
                                        self._parent_class._password = None
                                        # remove referer header prior to continuing.
                                        self._parent_class.remove_header('Referer')
                                        return False

                                else:
                                    print("ESP Client Login failed.")
                                    # clear password out of memory
                                    self._parent_class.email = None
                                    self._parent_class._password = None
                                    # remove referer header prior to continuing.
                                    self._parent_class.remove_header('Referer')
                                    return False

                            else:
                                print("ESP Client Choice failed.")
                                # clear password out of memory
                                self._parent_class.email = None
                                self._parent_class._password = None
                                # remove referer header prior to continuing.
                                self._parent_class.remove_header('Referer')
                                return False

                        # successful!
                        # clear password out of memory
                        self._parent_class._password = None
                        # remove referer header prior to continuing.
                        self._parent_class.remove_header('Referer')
                        return True

                    else:
                        print("Tenant detail retrieval failed.")
                        # clear password out of memory
                        self._parent_class.email = None
                        self._parent_class._password = None
                        # remove referer header prior to continuing.
                        self._parent_class.remove_header('Referer')
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

            # remove referer header prior to continuing.
            self._parent_class.remove_header('Referer')
        else:
            # log response when debug
            api_logger.debug("LOGIN_FAIL_RESPONSE = %s", json.dumps(response.cgx_content, indent=4))
            # print login error
            print('Login failed, please try again:', response.cgx_content)
            # Flush command-line entered login info if failure.
            self._parent_class.email = None
            self._parent_class.password = None

            # remove referer header prior to continuing.
            self._parent_class.remove_header('Referer')
        return False