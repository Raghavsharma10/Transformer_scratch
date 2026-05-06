def user_update(self, display_name=None, first_name=None, last_name=None,
                    email=None, password=None, current_password=None,
                    birth_date=None, gender=None, website=None, subdomain=None,
                    location=None, newsletter=None, primary_usage=None,
                    timezone=None):
        """
        user/update

        http://www.mediafire.com/developers/core_api/1.3/user/#update
        """
        return self.request("user/update", QueryParams({
            "display_name": display_name,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": password,
            "current_password": current_password,
            "birth_date": birth_date,
            "gender": gender,
            "website": website,
            "subdomain": subdomain,
            "location": location,
            "newsletter": newsletter,
            "primary_usage": primary_usage,
            "timezone": timezone
        }))