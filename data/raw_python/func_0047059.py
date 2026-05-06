def authentications_spec(self):
        """Spec for a group of authentication options"""
        return container_spec(authentication_objs.Authentication
              , dictof(string_spec(), set_options(
                  reading = optional_spec(authentication_spec())
                , writing = optional_spec(authentication_spec())
                )
              )
            )