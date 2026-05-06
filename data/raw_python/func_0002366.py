def _set_default_configuration_options(app):
        """
        Sets the default configuration options used by this extension
        """
        # Options for JWTs when the TOKEN_LOCATION is headers
        app.config.setdefault('JWT_HEADER_NAME', 'Authorization')
        app.config.setdefault('JWT_HEADER_TYPE', 'Bearer')

        # How long an a token created with 'create_jwt' will last before
        # it expires (when using the default jwt_data_callback function).
        app.config.setdefault('JWT_EXPIRES', datetime.timedelta(hours=1))

        # What algorithm to use to sign the token. See here for a list of options:
        # https://github.com/jpadilla/pyjwt/blob/master/jwt/api_jwt.py
        app.config.setdefault('JWT_ALGORITHM', 'HS256')

        # Key that acts as the identity for the JWT
        app.config.setdefault('JWT_IDENTITY_CLAIM', 'sub')

        # Expected value of the audience claim
        app.config.setdefault('JWT_DECODE_AUDIENCE', None)

        # Secret key to sign JWTs with. Only used if a symmetric algorithm is
        # used (such as the HS* algorithms).
        app.config.setdefault('JWT_SECRET_KEY', None)

        # Keys to sign JWTs with when use when using an asymmetric
        # (public/private key) algorithms, such as RS* or EC*
        app.config.setdefault('JWT_PRIVATE_KEY', None)
        app.config.setdefault('JWT_PUBLIC_KEY', None)