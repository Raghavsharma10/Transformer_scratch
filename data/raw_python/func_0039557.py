def web_services_from_str(
    list_splitter_fn=ujson.loads,
):
    """
    parameters:
        list_splitter_fn - a function that will take the json compatible string
            rerpesenting a list of mappings.
    """

    # -------------------------------------------------------------------------
    def class_list_converter(collector_services_str):
        """This function becomes the actual converter used by configman to
        take a string and convert it into the nested sequence of Namespaces,
        one for each class in the list.  It does this by creating a proxy
        class stuffed with its own 'required_config' that's dynamically
        generated."""
        if isinstance(collector_services_str, basestring):
            all_collector_services = list_splitter_fn(collector_services_str)
        else:
            raise TypeError('must be derivative of a basestring')

        # =====================================================================
        class InnerClassList(RequiredConfig):
            """This nested class is a proxy list for the classes.  It collects
            all the config requirements for the listed classes and places them
            each into their own Namespace.
            """
            # we're dynamically creating a class here.  The following block of
            # code is actually adding class level attributes to this new class

            # 1st requirement for configman
            required_config = Namespace()

            # to help the programmer know what Namespaces we added
            subordinate_namespace_names = []

            # for display
            original_input = collector_services_str.replace('\n', '\\n')

            # for each class in the class list
            service_list = []
            for namespace_index, collector_service_element in enumerate(
                all_collector_services
            ):
                service_name = collector_service_element['name']
                service_uri = collector_service_element['uri']
                service_implementation_class = class_converter(
                    collector_service_element['service_implementation_class']
                )

                service_list.append(
                    (
                        service_name,
                        service_uri,
                        service_implementation_class,
                    )
                )
                subordinate_namespace_names.append(service_name)
                # create the new Namespace
                required_config.namespace(service_name)
                a_class_namespace = required_config[service_name]
                a_class_namespace.add_option(
                    "service_implementation_class",
                    doc='fully qualified classname for a class that implements'
                        'the action associtated with the URI',
                    default=service_implementation_class,
                    from_string_converter=class_converter,
                    likely_to_be_changed=True,
                )
                a_class_namespace.add_option(
                    "uri",
                    doc='uri for this service',
                    default=service_uri,
                    likely_to_be_changed=True,
                )

            @classmethod
            def to_str(cls):
                """this method takes this inner class object and turns it back
                into the original string of classnames.  This is used
                primarily as for the output of the 'help' option"""
                return "'%s'" % cls.original_input

        return InnerClassList  # result of class_list_converter

    return class_list_converter