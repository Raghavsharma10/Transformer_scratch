def load_template(self, template_name, template_source=None,
                      template_path=None, **template_vars):
        """
        Will load a templated configuration on the device.

        :param cls: Instance of the driver class.
        :param template_name: Identifies the template name.
        :param template_source (optional): Custom config template rendered and loaded on device
        :param template_path (optional): Absolute path to directory for the configuration templates
        :param template_vars: Dictionary with arguments to be used when the template is rendered.
        :raise DriverTemplateNotImplemented: No template defined for the device type.
        :raise TemplateNotImplemented: The template specified in template_name does not exist in \
        the default path or in the custom path if any specified using parameter `template_path`.
        :raise TemplateRenderException: The template could not be rendered. Either the template \
        source does not have the right format, either the arguments in `template_vars` are not \
        properly specified.
        """
        return napalm_base.helpers.load_template(self,
                                                 template_name,
                                                 template_source=template_source,
                                                 template_path=template_path,
                                                 **template_vars)