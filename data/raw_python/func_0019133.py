def get(self, request, pk):
        """ Initialize the editing form

        1. Build opening_hours, a lookup dictionary to populate the form
           slots: keys are day numbers, values are lists of opening
           hours for that day.
        2. Build days, a list of days with 2 slot forms each.
        3. Build form initials for the 2 slots padding/trimming
           opening_hours to end up with exactly 2 slots even if it's
           just None values.
        """
        location = self.get_object()
        two_sets = False
        closed = None
        opening_hours = {}
        for o in OpeningHours.objects.filter(company=location):
            opening_hours.setdefault(o.weekday, []).append(o)
        days = []
        for day_no, day_name in WEEKDAYS:
            if day_no not in opening_hours.keys():
                if opening_hours:
                    closed = True
                ini1, ini2 = [None, None]
            else:
                closed = False
                ini = [{'opens': time_to_str(oh.from_hour),
                        'shuts': time_to_str(oh.to_hour)}
                       for oh in opening_hours[day_no]]
                ini += [None] * (2 - len(ini[:2]))  # pad
                ini1, ini2 = ini[:2]  # trim
                if ini2:
                    two_sets = True
            days.append({
                'name': day_name,
                'number': day_no,
                'slot1': Slot(prefix=self.form_prefix(day_no, 1), initial=ini1),
                'slot2': Slot(prefix=self.form_prefix(day_no, 2), initial=ini2),
                'closed': closed
            })
        return render(request, self.template_name, {
            'days': days,
            'two_sets': two_sets,
            'location': location,
        })