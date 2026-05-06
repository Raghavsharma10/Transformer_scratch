def _render_config(cls, dest, template_name, template_args):
        """
        Renders and writes a template_name to a dest given some template_args.

        This is for platform-specific configurations
        """
        template_args = template_args.copy()

        # Substitute values here
        pyversion = template_args['pyversion']
        template_args['pyversion'] = cls.PYVERSIONS[pyversion]

        template = template_env.get_template(template_name)
        contents = template.render(**template_args)
        _write_file(dest, contents)