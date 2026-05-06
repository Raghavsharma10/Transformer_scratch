def generate_timestamped_string(subject="test", number_of_random_chars=4):
    """
    Generate time-stamped string. Format as follows...

    `2013-01-31_14:12:23_SubjectString_a3Zg`


    Kwargs:
        subject (str): String to use as subject.
        number_of_random_chars (int) : Number of random characters to append.


    This method is helpful for creating unique names with timestamps in them so 
    when you have to troubleshoot an issue, the name is easier to find.::

        self.project_name = generate_timestamped_string("project")
        new_project_page.create_project(project_name)

    """
    random_str = generate_random_string(number_of_random_chars)
    timestamp = generate_timestamp()
    return u"{timestamp}_{subject}_{random_str}".format(timestamp=timestamp,
                                                        subject=subject,
                                                        random_str=random_str)