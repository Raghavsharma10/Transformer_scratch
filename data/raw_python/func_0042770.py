def execute(self, name):
        """Orchestrate the work of the generator plugin. It is split into multiple
        phases:
        * initializing
        * prompting
        * configuring
        * writing

        :param gen: the generator you want to run
        :return:
        """
        generator = self._get_generator(name)
        generator.initializing()
        answers = generator.prompting(create_store_prompt(name))
        # TODO
        # app.insight.track('yoyo', 'help', answers)
        generator.configuring(answers)
        generator.writing(answers)