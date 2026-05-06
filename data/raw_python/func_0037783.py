def render(self, tmpl_name, request_env):
        """
        Render the specified template and return the output.

        Args:
            tmpl_name (str): file name of the template
            request_env (dict): request environment


        Returns:
            str - the rendered template
        """
        return super(WebApplication, self).render(tmpl_name, request_env)