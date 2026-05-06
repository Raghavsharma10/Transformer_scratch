def table_api_put(self, *paths, **kparams):
        """ helper to make PUT /api/now/v1/table requests """
        url = self.flattened_params_url("/api/now/v1/table", *paths)

        # json.dumps(kparams) is the body of the put/post
        rjson = self.req("put", url, json.dumps(kparams)).text
        return json.loads(rjson)