def reinit_configurations(self, request):
        """ Re-initialize configuration for resource if it has been changed.

            This method should be called if resource consumption strategy was changed.
        """
        now = timezone.now()

        # Step 1. Collect all resources with changed configuration.
        changed_resources = []
        for resource_model in CostTrackingRegister.registered_resources:
            for resource in resource_model.objects.all():
                try:
                    pe = models.PriceEstimate.objects.get(scope=resource, month=now.month, year=now.year)
                except models.PriceEstimate.DoesNotExist:
                    changed_resources.append(resource)
                else:
                    new_configuration = CostTrackingRegister.get_configuration(resource)
                    if new_configuration != pe.consumption_details.configuration:
                        changed_resources.append(resource)

        # Step 2. Re-init configuration and recalculate estimate for changed resources.
        for resource in changed_resources:
            models.PriceEstimate.update_resource_estimate(resource, CostTrackingRegister.get_configuration(resource))

        message = _('Configuration was reinitialized for %(count)s resources') % {'count': len(changed_resources)}
        self.message_user(request, message)

        return redirect(reverse('admin:cost_tracking_defaultpricelistitem_changelist'))