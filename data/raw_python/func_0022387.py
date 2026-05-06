def interactive_tenant_update_vars(self):
        """
        Function to update the `cloudgenix.API` object with tenant login info. Run after login or client login.

        **Returns:** Boolean on success/failure,
        """
        api_logger.info('interactive_tenant_update_vars function:')
        tenant_resp = self._parent_class.get.tenants(self._parent_class.tenant_id)
        status = tenant_resp.cgx_status
        tenant_dict = tenant_resp.cgx_content

        if status:

            api_logger.debug("new tenant_dict: %s", tenant_dict)

            # Get Tenant info.
            self._parent_class.tenant_name = tenant_dict.get('name', self._parent_class.tenant_id)
            # is ESP/MSP?
            self._parent_class.is_esp = tenant_dict.get('is_esp')
            # grab tenant address for location.
            address_lookup = tenant_dict.get('address', None)
            if address_lookup:
                tenant_address = address_lookup.get('street', "") + ", "
                tenant_address += (str(address_lookup.get('street2', "")) + ", ")
                tenant_address += (str(address_lookup.get('city', "")) + ", ")
                tenant_address += (str(address_lookup.get('state', "")) + ", ")
                tenant_address += (str(address_lookup.get('post_code', "")) + ", ")
                tenant_address += (str(address_lookup.get('country', "")) + ", ")
            else:
                tenant_address = "Unknown"
            self._parent_class.address = tenant_address
            return True
        else:
            # update failed
            return False