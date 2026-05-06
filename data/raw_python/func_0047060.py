def wait_condition_spec(self):
        """Spec for a wait_condition block"""
        from harpoon.option_spec import image_objs
        formatted_string = formatted(string_spec(), formatter=MergedOptionStringFormatter)
        return create_spec(image_objs.WaitCondition
            , harpoon = formatted(overridden("{harpoon}"), formatter=MergedOptionStringFormatter)
            , timeout = defaulted(integer_spec(), 300)
            , wait_between_attempts = defaulted(float_spec(), 5)

            , greps = optional_spec(dictof(formatted_string, formatted_string))
            , command = optional_spec(listof(formatted_string))
            , port_open = optional_spec(listof(integer_spec()))
            , file_value = optional_spec(dictof(formatted_string, formatted_string))
            , curl_result = optional_spec(dictof(formatted_string, formatted_string))
            , file_exists = optional_spec(listof(formatted_string))
            )