def get_choices_for_models():
        """
        Get the select choices for models in optgroup mode
        :return:
        """
        result = {}

        apps = [item.model.split(' - ')[0] for item in TransModelLanguage.objects.all()]
        qs = ContentType.objects.exclude(model__contains='translation').filter(app_label__in=apps).order_by(
            'app_label', 'model'
        )
        for ct in qs:
            if not issubclass(ct.model_class(), TranslatableModel):
                continue
            if ct.app_label not in result:
                result[ct.app_label] = []
            result[ct.app_label].append((
                ct.id, str(ct.model_class()._meta.verbose_name_plural)
            ))
        choices = [(str(_('Todas las clases')), (('', _('Todas las clases')),))]
        for key, value in result.items():
            choices.append((key, tuple([it for it in sorted(value, key=lambda el: el[1])])))
        return choices