def access_elementusers(self, elementuser_id, access_id=None, tenant_id=None, api_version="v2.0"):
        """
        Get all accesses for a particular user

          **Parameters:**:

          - **elementuser_id**: Element User ID
          - **access_id**: (optional) Access ID
          - **tenant_id**: Tenant ID
          - **api_version**: API version to use (default v2.0)

        **Returns:** requests.Response object extended with cgx_status and cgx_content properties.
        """

        if tenant_id is None and self._parent_class.tenant_id:
            # Pull tenant_id from parent namespace cache.
            tenant_id = self._parent_class.tenant_id
        elif not tenant_id:
            # No value for tenant_id.
            raise TypeError("tenant_id is required but not set or cached.")
        cur_ctlr = self._parent_class.controller

        if not access_id:
            url = str(cur_ctlr) + "/{}/api/tenants/{}/elementusers/{}/access".format(api_version,
                                                                                     tenant_id,
                                                                                     elementuser_id)
        else:
            url = str(cur_ctlr) + "/{}/api/tenants/{}/elementusers/{}/access/{}".format(api_version,
                                                                                        tenant_id,
                                                                                        elementuser_id,
                                                                                        access_id)

        api_logger.debug("URL = %s", url)
        return self._parent_class.rest_call(url, "get")