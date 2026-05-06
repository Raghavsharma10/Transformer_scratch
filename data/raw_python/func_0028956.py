def deploy_custom_domain(awsclient, api_name, api_target_stage,
                         api_base_path, domain_name, route_53_record,
                         cert_name, cert_arn, hosted_zone_id, ensure_cname):
    """Add custom domain to your API.

    :param api_name:
    :param api_target_stage:
    :param api_base_path:
    :param domain_name:
    :param route_53_record:
    :param ssl_cert:
    :param cert_name:
    :param cert_arn:
    :param hosted_zone_id:
    :return: exit_code
    """
    api_base_path = _basepath_to_string_if_null(api_base_path)
    api = _api_by_name(awsclient, api_name)

    if not api:
        print("Api %s does not exist, aborting..." % api_name)
        # exit(1)
        return 1

    domain = _custom_domain_name_exists(awsclient, domain_name)

    if not domain:
        response = _create_custom_domain(awsclient, domain_name, cert_name, cert_arn)
        cloudfront_distribution = response['distributionDomainName']
    else:
        response = _update_custom_domain(awsclient, domain_name, cert_name, cert_arn)
        cloudfront_distribution = response['distributionDomainName']

    if _base_path_mapping_exists(awsclient, domain_name, api_base_path):
        _ensure_correct_base_path_mapping(awsclient, domain_name,
                                          api_base_path, api['id'],
                                          api_target_stage)
    else:
        _create_base_path_mapping(awsclient, domain_name, api_base_path,
                                  api_target_stage, api['id'])

    if ensure_cname:
        record_exists, record_correct = \
            _record_exists_and_correct(awsclient, hosted_zone_id,
                                       route_53_record,
                                       cloudfront_distribution)
        if record_correct:
            print('Route53 record correctly set: %s --> %s' % (route_53_record,
                                                               cloudfront_distribution))
        else:
            _ensure_correct_route_53_record(awsclient, hosted_zone_id,
                                            record_name=route_53_record,
                                            record_value=cloudfront_distribution)
            print('Route53 record set: %s --> %s' % (route_53_record,
                                                     cloudfront_distribution))
    else:
        print('Skipping creating and checking DNS record')

    return 0