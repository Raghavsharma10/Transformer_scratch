def request(self, method, suffix, data):
        """
        :param method: str, http method ["GET","POST","PUT"]
        :param suffix: the url suffix
        :param data:
        :return:
        """
        url = self.site_url + suffix
        response = self.session.request(method, url, data=data)

        if response.status_code == 200:
            json_obj = response.json()

            if isinstance(json_obj, dict) and json_obj.get("error_code"):

                raise WeiboOauth2Error(
                    json_obj.get("error_code"),
                    json_obj.get("error"),
                    json_obj.get('error_description')
                )
            else:
                return json_obj
        else:
            raise WeiboRequestError(
                "Weibo API request error: status code: {code} url:{url} ->"
                " method:{method}: data={data}".format(
                    code=response.status_code,
                    url=response.url,
                    method=method,
                    data=data
                )
            )