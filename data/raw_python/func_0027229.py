def list(self, request, *args, **kwargs):
        """
        To get a list of supported resources' actions, run **OPTIONS** against
        */api/<resource_url>/* as an authenticated user.

        It is possible to filter and order by resource-specific fields, but this filters will be applied only to
        resources that support such filtering. For example it is possible to sort resource by ?o=ram, but SugarCRM crms
        will ignore this ordering, because they do not support such option.

        Filter resources by type or category
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        There are two query argument to select resources by their type.

        - Specify explicitly list of resource types, for example:

          /api/<resource_endpoint>/?resource_type=DigitalOcean.Droplet&resource_type=OpenStack.Instance

        - Specify category, one of vms, apps, private_clouds or storages for example:

          /api/<resource_endpoint>/?category=vms

        Filtering by monitoring fields
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Resources may have SLA attached to it. Example rendering of SLA:

        .. code-block:: javascript

            "sla": {
                "value": 95.0
                "agreed_value": 99.0,
                "period": "2016-03"
            }

        You may filter or order resources by SLA. Default period is current year and month.

        - Example query for filtering list of resources by actual SLA:

          /api/<resource_endpoint>/?actual_sla=90&period=2016-02

        - Warning! If resource does not have SLA attached to it, it is not included in ordered response.
          Example query for ordering list of resources by actual SLA:

          /api/<resource_endpoint>/?o=actual_sla&period=2016-02

        Service list is displaying current SLAs for each of the items. By default,
        SLA period is set to the current month. To change the period pass it as a query argument:

        - ?period=YYYY-MM - return a list with SLAs for a given month
        - ?period=YYYY - return a list with SLAs for a given year

        In all cases all currently running resources are returned, if SLA for the given period is
        not known or not present, it will be shown as **null** in the response.

        Resources may have monitoring items attached to it. Example rendering of monitoring items:

        .. code-block:: javascript

            "monitoring_items": {
               "application_state": 1
            }

        You may filter or order resources by monitoring item.

        - Example query for filtering list of resources by installation state:

          /api/<resource_endpoint>/?monitoring__installation_state=1

        - Warning! If resource does not have monitoring item attached to it, it is not included in ordered response.
          Example query for ordering list of resources by installation state:

          /api/<resource_endpoint>/?o=monitoring__installation_state

        Filtering by tags
        ^^^^^^^^^^^^^^^^^

        Resource may have tags attached to it. Example of tags rendering:

        .. code-block:: javascript

            "tags": [
                "license-os:centos7",
                "os-family:linux",
                "license-application:postgresql",
                "support:premium"
            ]

        Tags filtering:

         - ?tag=IaaS - filter by full tag name, using method OR. Can be list.
         - ?rtag=os-family:linux - filter by full tag name, using AND method. Can be list.
         - ?tag__license-os=centos7 - filter by tags with particular prefix.

        Tags ordering:

         - ?o=tag__license-os - order by tag with particular prefix. Instances without given tag will not be returned.
        """

        return super(ResourceSummaryViewSet, self).list(request, *args, **kwargs)