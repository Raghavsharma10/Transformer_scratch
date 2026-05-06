def to_internal_value(self, data):
        """
        Restores model instance from its url
        """
        if not data:
            return None
        request = self._get_request()
        user = request.user
        try:
            obj = core_utils.instance_from_url(data, user=user)
            model = obj.__class__
        except ValueError:
            raise serializers.ValidationError(_('URL is invalid: %s.') % data)
        except (Resolver404, AttributeError, MultipleObjectsReturned, ObjectDoesNotExist):
            raise serializers.ValidationError(_("Can't restore object from url: %s") % data)
        if model not in self.related_models:
            raise serializers.ValidationError(_('%s object does not support such relationship.') % six.text_type(obj))
        return obj