def always_fails(
        self,
        work_dict):
    """always_fails

    :param work_dict: dictionary for key/values
    """

    label = "always_fails"

    log.info(("task - {} - start "
              "work_dict={}")
             .format(label,
                     work_dict))

    raise Exception(
            work_dict.get(
                "test_failure",
                "simulating a failure"))

    log.info(("task - {} - done")
             .format(label))

    return True