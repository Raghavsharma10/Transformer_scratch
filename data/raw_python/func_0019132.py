def post(self, request, pk):
        """ Clean the data and save opening hours in the database.
        Old opening hours are purged before new ones are saved.
        """
        location = self.get_object()
        # open days, disabled widget data won't make it into request.POST
        present_prefixes = [x.split('-')[0] for x in request.POST.keys()]
        day_forms = OrderedDict()
        for day_no, day_name in WEEKDAYS:
            for slot_no in (1, 2):
                prefix = self.form_prefix(day_no, slot_no)
                # skip closed day as it would be invalid form due to no data
                if prefix not in present_prefixes:
                    continue
                day_forms[prefix] = (day_no, Slot(request.POST, prefix=prefix))

        if all([day_form[1].is_valid() for pre, day_form in day_forms.items()]):
            OpeningHours.objects.filter(company=location).delete()
            for prefix, day_form in day_forms.items():
                day, form = day_form
                opens, shuts = [str_to_time(form.cleaned_data[x])
                                for x in ('opens', 'shuts')]
                if opens != shuts:
                    OpeningHours(from_hour=opens, to_hour=shuts,
                                 company=location, weekday=day).save()
        return redirect(request.path_info)