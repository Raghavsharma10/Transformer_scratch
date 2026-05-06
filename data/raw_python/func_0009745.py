def check_implicit_rules(self):
        """ if you are using flask classy are using the standard index,new,put,post, etc ... type routes, we will
            automatically check the permissions for you
        """
        if not self.request_is_managed_by_flask_classy():
            return

        if self.method_is_explictly_overwritten():
            return

        class_name, action = request.endpoint.split(':')
        clazz = [classy_class for classy_class in self.flask_classy_classes if classy_class.__name__ == class_name][0]
        Condition(action, clazz.__target_model__).test()