def get_form_label(self, request=None, obj=None, model=None, form=None):
        """Returns a customized form label, if condition is met,
        otherwise returns the default form label.

        * condition is an instance of CustomLabelCondition.
        """
        label = form.base_fields[self.field].label
        condition = self.condition_cls(request=request, obj=obj, model=model)
        if condition.check():
            additional_opts = condition.get_additional_options(
                request=request, obj=obj, model=model
            )
            visit_datetime = ""
            if obj:
                visit_datetime = getattr(
                    obj, obj.visit_model_attr()
                ).report_datetime.strftime("%B %Y")
            try:
                label = self.custom_label.format(
                    appointment=condition.appointment,
                    previous_appointment=condition.previous_appointment,
                    previous_obj=condition.previous_obj,
                    previous_visit=condition.previous_visit,
                    visit_datetime=visit_datetime,
                    **additional_opts,
                )
            except KeyError as e:
                raise CustomFormLabelError(
                    f"Custom label template has invalid keys. See {label}. Got {e}."
                )
        return label