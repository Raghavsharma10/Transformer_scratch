def process_models(attrs, base_model_class):
        """
        Attach default fields and meta options to models
        """
        attrs.update(base_model_class._DEFAULT_BASE_FIELDS)
        attrs['_instance_registry'] = set()
        attrs['_is_unpermitted_fields_set'] = False
        attrs['save_meta_data'] = None
        attrs['_pre_save_hook_called'] = False
        attrs['_post_save_hook_called'] = False
        DEFAULT_META = {'bucket_type': settings.DEFAULT_BUCKET_TYPE,
                        'field_permissions': {},
                        'app': 'main',
                        'list_fields': [],
                        'list_filters': [],
                        'search_fields': [],
                        'fake_model': False,
                        }
        if 'Meta' not in attrs:
            attrs['Meta'] = type('Meta', (object,), DEFAULT_META)
        else:
            for k, v in DEFAULT_META.items():
                if k not in attrs['Meta'].__dict__:
                    setattr(attrs['Meta'], k, v)