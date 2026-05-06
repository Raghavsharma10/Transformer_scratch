def _handler_response(self, response, data=None):
        """
        error code response:
        {
            "request": "/statuses/home_timeline.json",
            "error_code": "20502",
            "error": "Need you follow uid."
        }
        :param response:
        :return:
        """
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and data.get("error_code"):

                raise WeiboAPIError(data.get("request"), data.get("error_code"), data.get("error"))
            else:
                return data
        else:
            raise WeiboRequestError(
                "Weibo API request error: status code: {code} url:{url} ->"
                " method:{method}: data={data}".format(
                    code=response.status_code,
                    url=response.url,
                    method=response.request.method,
                    data=data
                )
            )