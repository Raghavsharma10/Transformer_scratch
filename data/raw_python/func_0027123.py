def init_registered(self, request):
        """ Create default price list items for each registered resource. """
        created_items = models.DefaultPriceListItem.init_from_registered_resources()

        if created_items:
            message = ungettext(
                _('Price item was created: %s.') % created_items[0].name,
                _('Price items were created: %s.') % ', '.join(item.name for item in created_items),
                len(created_items)
            )
            self.message_user(request, message)
        else:
            self.message_user(request, _('Price items for all registered resources have been updated.'))

        return redirect(reverse('admin:cost_tracking_defaultpricelistitem_changelist'))