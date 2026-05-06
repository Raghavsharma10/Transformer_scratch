def check_expression(testing_framework, expression_dict):
    """
    >>> class mock_framework:
    ...   def assertIn(self, item, list, msg="Failed asserting item is in list"):
    ...     if item not in list: raise Exception(msg)
    ...   def assertTrue(self, value, msg="Failed asserting true"):
    ...     if not value: raise Exception(msg)
    ...   def assertFalse(self, value, msg): self.assertTrue(not value, msg)
    >>> check_expression(mock_framework(),
    ...   {'class': {'group' :{'Matches': " 0 | 1", 'Non-Matches': "2 | 0 2", 'Expression': "[0-1]"}}})
    """
    expression_sub = get_expression_sub()
    
    for expression_type_name, expression_type in expression_dict.items():
        for name, expression_object in expression_type.items():
            if 'Matches' in expression_object.keys():
                for test in expression_object['Matches'].split('|'):
                    # Substitute and check to make sure that the entire string matches
                    result = expression_sub(expression_object['Expression'], '', test.strip()) == ''
                    testing_framework.assertTrue(result, match_error_msg.format(expression_type_name, name, test))
            if 'Non-Matches' in expression_object.keys():
                for test in expression_object['Non-Matches'].split('|'):
                    result = expression_sub(expression_object['Expression'], '', test.strip()) == ''
                    testing_framework.assertFalse(result, non_match_error_msg.format(expression_type_name, name, test))