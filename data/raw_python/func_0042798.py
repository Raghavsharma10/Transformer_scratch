def labels(self):
        """
        Get field label for fields
        """

        if type(self.object_list) == type([]):
            model = self.formset.model
        else:
            model = self.object_list.model

        for field in self.visible_fields:
            name = None
            if self.formset:
                f = self.formset.empty_form.fields.get(field, None)
                if f:
                    name = f.label

            if name is None:
                name = label_for_field(field, model)

            if name == model._meta.verbose_name:
                name = self.model_name and self.model_name or \
                            model._meta.verbose_name

            stype = None
            cur_sorted = False

            sortable = False

            if self.order_type:
                sortable = get_sort_field(field, model)
                stype = self.ASC

                # change order_type so that next sorting on the same
                # field will give reversed results
                if sortable and field == self.sort_field:
                    cur_sorted = True
                    if self.order_type == self.ASC:
                        stype = self.DESC
                    elif self.order_type == self.DESC:
                        stype = self.ASC
                    else:
                        stype = self.ASC


            yield AdminListLabel(name, field, stype, cur_sorted, bool(sortable))