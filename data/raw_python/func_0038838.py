def create_translation_tasks(self, instance):
        """
        Creates the translations tasks from the instance and its translatable children

        :param instance:
        :return:
        """

        langs = self.get_languages()

        result = []
        # get the previous and actual values
        # in case it's and "add" operation previous values will be empty
        previous_values, actual_values = self.get_previous_and_current_values(instance)

        # extract the differences
        differences = self.extract_diferences(previous_values, actual_values)
        self.log('\nprev: {}\nactu:{}\ndiff:{}'.format(previous_values, actual_values, differences))
        if len(differences) > 0:
            # there are differences in the main model, so we create the tasks for it
            result += self.create_from_item(langs, instance.master, differences, trans_instance=self.instance)
        else:
            # no differences so we do nothing to the main model
            self.log('No differences we do nothing CREATE {}:{}'.format(self.master_class, instance.language_code))
        return result