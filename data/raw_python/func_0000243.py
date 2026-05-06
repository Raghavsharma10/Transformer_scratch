def prepare_env(org):
    """ Example shows how to configure environment from scratch """

    # Add services
    key_service = org.service(type='builtin:cobalt_secure_store', name='Keystore')
    wf_service = org.service(type='builtin:workflow_service', name='Workflow', parameters='{}')

    # Add services to environment
    env = org.environment(name='default')
    env.clean()
    env.add_service(key_service)
    env.add_service(wf_service)
    env.add_policy(
        {"action": "provisionVms",
         "parameter": "publicKeyId",
         "value": key_service.regenerate()['id']})

    # Add cloud provider account
    access = {
      "provider": "aws-ec2",
      "usedEnvironments": [],
      "ec2SecurityGroup": "default",
      "providerCopy": "aws-ec2",
      "name": "test-provider",
      "jcloudsIdentity": KEY,
      "jcloudsCredential": SECRET_KEY,
      "jcloudsRegions": "us-east-1"
    }
    prov = org.provider(access)
    env.add_provider(prov)
    return org.organizationId