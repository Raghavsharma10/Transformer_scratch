def store_custom_data(self, ns, data, user_id):
        """
        Store custom data.

        Store arbitrary user data as JSON.
        
        Arbitrary JSON data can be stored for a User.
        A typical scenario would be an external site/service that registers users in Canvas
        and wants to capture additional info about them.  The part of the URL that follows
        +/custom_data/+ defines the scope of the request, and it reflects the structure of
        the JSON data to be stored or retrieved.
        
        The value +self+ may be used for +user_id+ to store data associated with the calling user.
        In order to access another user's custom data, you must be an account administrator with
        permission to manage users.
        
        A namespace parameter, +ns+, is used to prevent custom_data collisions between
        different apps.  This parameter is required for all custom_data requests.
        
        A request with Content-Type multipart/form-data or Content-Type
        application/x-www-form-urlencoded can only be used to store strings.
        
        Example PUT with multipart/form-data data:
          curl 'https://<canvas>/api/v1/users/<user_id>/custom_data/telephone' \
            -X PUT \
            -F 'ns=com.my-organization.canvas-app' \
            -F 'data=555-1234' \
            -H 'Authorization: Bearer <token>'
        
        Response:
          !!!javascript
          {
            "data": "555-1234"
          }
        
        Subscopes (or, generated scopes) can also be specified by passing values to
        +data+[+subscope+].
        
        Example PUT specifying subscopes:
          curl 'https://<canvas>/api/v1/users/<user_id>/custom_data/body/measurements' \
            -X PUT \
            -F 'ns=com.my-organization.canvas-app' \
            -F 'data[waist]=32in' \
            -F 'data[inseam]=34in' \
            -F 'data[chest]=40in' \
            -H 'Authorization: Bearer <token>'
        
        Response:
          !!!javascript
          {
            "data": {
              "chest": "40in",
              "waist": "32in",
              "inseam": "34in"
            }
          }
        
        Following such a request, subsets of the stored data to be retrieved directly from a subscope.
        
        Example {api:UsersController#get_custom_data GET} from a generated scope
          curl 'https://<canvas>/api/v1/users/<user_id>/custom_data/body/measurements/chest' \
            -X GET \
            -F 'ns=com.my-organization.canvas-app' \
            -H 'Authorization: Bearer <token>'
        
        Response:
          !!!javascript
          {
            "data": "40in"
          }
        
        If you want to store more than just strings (i.e. numbers, arrays, hashes, true, false,
        and/or null), you must make a request with Content-Type application/json as in the following
        example.
        
        Example PUT with JSON data:
          curl 'https://<canvas>/api/v1/users/<user_id>/custom_data' \
            -H 'Content-Type: application/json' \
            -X PUT \
            -d '{
                  "ns": "com.my-organization.canvas-app",
                  "data": {
                    "a-number": 6.02e23,
                    "a-bool": true,
                    "a-string": "true",
                    "a-hash": {"a": {"b": "ohai"}},
                    "an-array": [1, "two", null, false]
                  }
                }' \
            -H 'Authorization: Bearer <token>'
        
        Response:
          !!!javascript
          {
            "data": {
              "a-number": 6.02e+23,
              "a-bool": true,
              "a-string": "true",
              "a-hash": {
                "a": {
                  "b": "ohai"
                }
              },
              "an-array": [1, "two", null, false]
            }
          }
        
        If the data is an Object (as it is in the above example), then subsets of the data can
        be accessed by including the object's (possibly nested) keys in the scope of a GET request.
        
        Example {api:UsersController#get_custom_data GET} with a generated scope:
          curl 'https://<canvas>/api/v1/users/<user_id>/custom_data/a-hash/a/b' \
            -X GET \
            -F 'ns=com.my-organization.canvas-app' \
            -H 'Authorization: Bearer <token>'
        
        Response:
          !!!javascript
          {
            "data": "ohai"
          }
        
        
        On success, this endpoint returns an object containing the data that was stored.
        
        Responds with status code 200 if the scope already contained data, and it was overwritten
        by the data specified in the request.
        
        Responds with status code 201 if the scope was previously empty, and the data specified
        in the request was successfully stored there.
        
        Responds with status code 400 if the namespace parameter, +ns+, is missing or invalid, or if
        the +data+ parameter is missing.
        
        Responds with status code 409 if the requested scope caused a conflict and data was not stored.
        This happens when storing data at the requested scope would cause data at an outer scope
        to be lost.  e.g., if +/custom_data+ was +{"fashion_app": {"hair": "blonde"}}+, but
        you tried to +`PUT /custom_data/fashion_app/hair/style -F data=buzz`+, then for the request
        to succeed,the value of +/custom_data/fashion_app/hair+ would have to become a hash, and its
        old string value would be lost.  In this situation, an error object is returned with the
        following format:
        
          !!!javascript
          {
            "message": "write conflict for custom_data hash",
            "conflict_scope": "fashion_app/hair",
            "type_at_conflict": "String",
            "value_at_conflict": "blonde"
          }
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - ns
        """The namespace under which to store the data.  This should be something other
        Canvas API apps aren't likely to use, such as a reverse DNS for your organization."""
        data["ns"] = ns

        # REQUIRED - data
        """The data you want to store for the user, at the specified scope.  If the data is
        composed of (possibly nested) JSON objects, scopes will be generated for the (nested)
        keys (see examples)."""
        data["data"] = data

        self.logger.debug("PUT /api/v1/users/{user_id}/custom_data with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/{user_id}/custom_data".format(**path), data=data, params=params, no_data=True)