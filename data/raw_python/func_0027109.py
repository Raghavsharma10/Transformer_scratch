def get_for_resource(resource):
        """ Get list of all price list items that should be used for resource.

            If price list item is defined for service - return it, otherwise -
            return default price list item.
        """
        resource_content_type = ContentType.objects.get_for_model(resource)
        default_items = set(DefaultPriceListItem.objects.filter(resource_content_type=resource_content_type))
        service = resource.service_project_link.service
        items = set(PriceListItem.objects.filter(
            default_price_list_item__in=default_items, service=service).select_related('default_price_list_item'))
        rewrited_defaults = set([i.default_price_list_item for i in items])
        return items | (default_items - rewrited_defaults)