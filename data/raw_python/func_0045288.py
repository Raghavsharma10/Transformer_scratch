def table_api_get(self, *paths, **kparams):
        """ helper to make GET /api/now/v1/table requests """
        url = self.flattened_params_url("/api/now/v1/table", *paths, **kparams)
        rjson = self.req("get", url).text
        return json.loads(rjson)