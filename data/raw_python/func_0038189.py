def create_order_classes(model_label, order_field_names):
    """
    Create order model and admin class.
    Add order model to order.models module and register admin class.
    Connect ordered_objects manager to related model.
    """
    # Seperate model_label into parts.
    labels = resolve_labels(model_label)
    # Get model class for model_label string.
    model = get_model(labels['app'], labels['model'])

    # Dynamic Classes
    class OrderItemBase(models.Model):
        """
        Dynamic order class base.
        """
        item = models.ForeignKey(model_label)
        timestamp = models.DateTimeField(auto_now=True)

        class Meta:
            abstract = True
            app_label = 'order'

    class Admin(admin.ModelAdmin):
        """
        Dynamic order admin class.
        """
        list_display = ('item_link',) + tuple(order_field_names)
        list_editable = order_field_names

        def get_model_perms(self, request):
            """
            Return empty perms dict thus hiding the model from admin index.
            """
            return {}

        @csrf_protect_m
        def changelist_view(self, request, extra_context=None):
            list_url = reverse('admin:%s_%s_changelist' % (labels['app'], \
                    labels['model'].lower()))
            add_url = reverse('admin:%s_%s_add' % (labels['app'], \
                    labels['model'].lower()))

            result = super(Admin, self).changelist_view(
                request,
                extra_context={
                    'add_url': add_url,
                    'list_url': list_url,
                    'related_opts': model._meta,
                }
            )

            # XXX: Sanitize order on list save.
            # if (request.method == "POST" and self.list_editable and \
            #        '_save' in request.POST):
            #    sanitize_order(self.model)
            return result

        def item_link(self, obj):
            url = reverse('admin:%s_%s_change' % (labels['app'], \
                    labels['model'].lower()), args=(obj.item.id,))
            return '<a href="%s">%s</a>' % (url, str(obj.item))
        item_link.allow_tags = True
        item_link.short_description = 'Item'

    # Set up a dictionary to simulate declarations within a class.
    attrs = {
        '__module__': 'order.models',
    }

    # Create provided order fields and add to attrs.
    fields = {}
    for field in order_field_names:
        fields[field] = models.IntegerField()
    attrs.update(fields)

    # Create the class which automatically triggers Django model processing.
    order_item_class_name = resolve_order_item_class_name(labels)
    order_model = type(order_item_class_name, (OrderItemBase, ), attrs)

    # Register admin model.
    admin.site.register(order_model, Admin)

    # Add user_order_by method to base QuerySet.
    from order import managers
    setattr(QuerySet, 'user_order_by', managers.user_order_by)

    # Return created model class.
    return order_model