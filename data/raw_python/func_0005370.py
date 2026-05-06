def dl_hosted(
        self,
        token: dict = None,
        resource_link: dict = None,
        encode_clean: bool = 1,
        proxy_url: str = None,
        prot: str = "https",
    ) -> tuple:
        """Download hosted resource.

        :param str token: API auth token
        :param dict resource_link: link dictionary
        :param bool encode_clean: option to ensure a clean filename and avoid OS errors
        :param str proxy_url: proxy to use to download
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).

        Example of resource_link dict:

        .. code-block:: json

            {
            "_id": "g8h9i0j11k12l13m14n15o16p17Q18rS",
            "type": "hosted",
            "title": "label_of_hosted_file.zip",
            "url": "/resources/1a2b3c4d5e6f7g8h9i0j11k12l13m14n/links/g8h9i0j11k12l13m14n15o16p17Q18rS.bin",
            "kind": "data",
            "actions": ["download", ],
            "size": "2253029",
            }

        """
        # check resource link parameter type
        if not isinstance(resource_link, dict):
            raise TypeError("Resource link expects a dictionary.")
        else:
            pass
        # check resource link type
        if not resource_link.get("type") == "hosted":
            raise ValueError(
                "Resource link passed is not a hosted one: {}".format(
                    resource_link.get("type")
                )
            )
        else:
            pass

        # handling request parameters
        payload = {"proxyUrl": proxy_url}

        # prepare URL request
        hosted_url = "{}://v1.{}.isogeo.com/{}".format(
            prot, self.api_url, resource_link.get("url")
        )

        # send stream request
        hosted_req = self.get(
            hosted_url,
            headers=self.header,
            stream=True,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )
        # quick check
        req_check = checker.check_api_response(hosted_req)
        if not req_check:
            raise ConnectionError(req_check[1])
        else:
            pass

        # get filename from header
        content_disposition = hosted_req.headers.get("Content-Disposition")
        if content_disposition:
            filename = re.findall("filename=(.+)", content_disposition)[0]
        else:
            filename = resource_link.get("title")

        # remove special characters
        if encode_clean:
            filename = utils.encoded_words_to_text(filename)
            filename = re.sub(r"[^\w\-_\. ]", "", filename)

        # well-formed size
        in_size = resource_link.get("size")
        for size_cat in ("octets", "Ko", "Mo", "Go"):
            if in_size < 1024.0:
                out_size = "%3.1f %s" % (in_size, size_cat)
            in_size /= 1024.0

        out_size = "%3.1f %s" % (in_size, " To")

        # end of method
        return (hosted_req, filename, out_size)