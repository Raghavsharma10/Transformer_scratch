def create_service(cluster=None, serviceName=None, taskDefinition=None, loadBalancers=None, desiredCount=None, clientToken=None, role=None, deploymentConfiguration=None, placementConstraints=None, placementStrategy=None):
    """
    Runs and maintains a desired number of tasks from a specified task definition. If the number of tasks running in a service drops below desiredCount , Amazon ECS spawns another copy of the task in the specified cluster. To update an existing service, see  UpdateService .
    In addition to maintaining the desired count of tasks in your service, you can optionally run your service behind a load balancer. The load balancer distributes traffic across the tasks that are associated with the service. For more information, see Service Load Balancing in the Amazon EC2 Container Service Developer Guide .
    You can optionally specify a deployment configuration for your service. During a deployment (which is triggered by changing the task definition or the desired count of a service with an  UpdateService operation), the service scheduler uses the minimumHealthyPercent and maximumPercent parameters to determine the deployment strategy.
    The minimumHealthyPercent represents a lower limit on the number of your service's tasks that must remain in the RUNNING state during a deployment, as a percentage of the desiredCount (rounded up to the nearest integer). This parameter enables you to deploy without using additional cluster capacity. For example, if your service has a desiredCount of four tasks and a minimumHealthyPercent of 50%, the scheduler can stop two existing tasks to free up cluster capacity before starting two new tasks. Tasks for services that do not use a load balancer are considered healthy if they are in the RUNNING state. Tasks for services that do use a load balancer are considered healthy if they are in the RUNNING state and the container instance they are hosted on is reported as healthy by the load balancer. The default value for minimumHealthyPercent is 50% in the console and 100% for the AWS CLI, the AWS SDKs, and the APIs.
    The maximumPercent parameter represents an upper limit on the number of your service's tasks that are allowed in the RUNNING or PENDING state during a deployment, as a percentage of the desiredCount (rounded down to the nearest integer). This parameter enables you to define the deployment batch size. For example, if your service has a desiredCount of four tasks and a maximumPercent value of 200%, the scheduler can start four new tasks before stopping the four older tasks (provided that the cluster resources required to do this are available). The default value for maximumPercent is 200%.
    When the service scheduler launches new tasks, it determines task placement in your cluster using the following logic:
    See also: AWS API Documentation
    
    Examples
    This example creates a service in your default region called ecs-simple-service. The service uses the hello_world task definition and it maintains 10 copies of that task.
    Expected Output:
    This example creates a service in your default region called ecs-simple-service-elb. The service uses the ecs-demo task definition and it maintains 10 copies of that task. You must reference an existing load balancer in the same region by its name.
    Expected Output:
    
    :example: response = client.create_service(
        cluster='string',
        serviceName='string',
        taskDefinition='string',
        loadBalancers=[
            {
                'targetGroupArn': 'string',
                'loadBalancerName': 'string',
                'containerName': 'string',
                'containerPort': 123
            },
        ],
        desiredCount=123,
        clientToken='string',
        role='string',
        deploymentConfiguration={
            'maximumPercent': 123,
            'minimumHealthyPercent': 123
        },
        placementConstraints=[
            {
                'type': 'distinctInstance'|'memberOf',
                'expression': 'string'
            },
        ],
        placementStrategy=[
            {
                'type': 'random'|'spread'|'binpack',
                'field': 'string'
            },
        ]
    )
    
    
    :type cluster: string
    :param cluster: The short name or full Amazon Resource Name (ARN) of the cluster on which to run your service. If you do not specify a cluster, the default cluster is assumed.

    :type serviceName: string
    :param serviceName: [REQUIRED]
            The name of your service. Up to 255 letters (uppercase and lowercase), numbers, hyphens, and underscores are allowed. Service names must be unique within a cluster, but you can have similarly named services in multiple clusters within a region or across multiple regions.
            

    :type taskDefinition: string
    :param taskDefinition: [REQUIRED]
            The family and revision (family:revision ) or full Amazon Resource Name (ARN) of the task definition to run in your service. If a revision is not specified, the latest ACTIVE revision is used.
            

    :type loadBalancers: list
    :param loadBalancers: A load balancer object representing the load balancer to use with your service. Currently, you are limited to one load balancer or target group per service. After you create a service, the load balancer name or target group ARN, container name, and container port specified in the service definition are immutable.
            For Elastic Load Balancing Classic load balancers, this object must contain the load balancer name, the container name (as it appears in a container definition), and the container port to access from the load balancer. When a task from this service is placed on a container instance, the container instance is registered with the load balancer specified here.
            For Elastic Load Balancing Application load balancers, this object must contain the load balancer target group ARN, the container name (as it appears in a container definition), and the container port to access from the load balancer. When a task from this service is placed on a container instance, the container instance and port combination is registered as a target in the target group specified here.
            (dict) --Details on a load balancer that is used with a service.
            targetGroupArn (string) --The full Amazon Resource Name (ARN) of the Elastic Load Balancing target group associated with a service.
            loadBalancerName (string) --The name of a Classic load balancer.
            containerName (string) --The name of the container (as it appears in a container definition) to associate with the load balancer.
            containerPort (integer) --The port on the container to associate with the load balancer. This port must correspond to a containerPort in the service's task definition. Your container instances must allow ingress traffic on the hostPort of the port mapping.
            
            

    :type desiredCount: integer
    :param desiredCount: [REQUIRED]
            The number of instantiations of the specified task definition to place and keep running on your cluster.
            

    :type clientToken: string
    :param clientToken: Unique, case-sensitive identifier you provide to ensure the idempotency of the request. Up to 32 ASCII characters are allowed.

    :type role: string
    :param role: The name or full Amazon Resource Name (ARN) of the IAM role that allows Amazon ECS to make calls to your load balancer on your behalf. This parameter is required if you are using a load balancer with your service. If you specify the role parameter, you must also specify a load balancer object with the loadBalancers parameter.
            If your specified role has a path other than / , then you must either specify the full role ARN (this is recommended) or prefix the role name with the path. For example, if a role with the name bar has a path of /foo/ then you would specify /foo/bar as the role name. For more information, see Friendly Names and Paths in the IAM User Guide .
            

    :type deploymentConfiguration: dict
    :param deploymentConfiguration: Optional deployment parameters that control how many tasks run during the deployment and the ordering of stopping and starting tasks.
            maximumPercent (integer) --The upper limit (as a percentage of the service's desiredCount ) of the number of tasks that are allowed in the RUNNING or PENDING state in a service during a deployment. The maximum number of tasks during a deployment is the desiredCount multiplied by maximumPercent /100, rounded down to the nearest integer value.
            minimumHealthyPercent (integer) --The lower limit (as a percentage of the service's desiredCount ) of the number of running tasks that must remain in the RUNNING state in a service during a deployment. The minimum healthy tasks during a deployment is the desiredCount multiplied by minimumHealthyPercent /100, rounded up to the nearest integer value.
            

    :type placementConstraints: list
    :param placementConstraints: An array of placement constraint objects to use for tasks in your service. You can specify a maximum of 10 constraints per task (this limit includes constraints in the task definition and those specified at run time).
            (dict) --An object representing a constraint on task placement. For more information, see Task Placement Constraints in the Amazon EC2 Container Service Developer Guide .
            type (string) --The type of constraint. Use distinctInstance to ensure that each task in a particular group is running on a different container instance. Use memberOf to restrict selection to a group of valid candidates. Note that distinctInstance is not supported in task definitions.
            expression (string) --A cluster query language expression to apply to the constraint. Note you cannot specify an expression if the constraint type is distinctInstance . For more information, see Cluster Query Language in the Amazon EC2 Container Service Developer Guide .
            
            

    :type placementStrategy: list
    :param placementStrategy: The placement strategy objects to use for tasks in your service. You can specify a maximum of 5 strategy rules per service.
            (dict) --The task placement strategy for a task or service. For more information, see Task Placement Strategies in the Amazon EC2 Container Service Developer Guide .
            type (string) --The type of placement strategy. The random placement strategy randomly places tasks on available candidates. The spread placement strategy spreads placement across available candidates evenly based on the field parameter. The binpack strategy places tasks on available candidates that have the least available amount of the resource that is specified with the field parameter. For example, if you binpack on memory, a task is placed on the instance with the least amount of remaining memory (but still enough to run the task).
            field (string) --The field to apply the placement strategy against. For the spread placement strategy, valid values are instanceId (or host , which has the same effect), or any platform or custom attribute that is applied to a container instance, such as attribute:ecs.availability-zone . For the binpack placement strategy, valid values are cpu and memory . For the random placement strategy, this field is not used.
            
            

    :rtype: dict
    :return: {
        'service': {
            'serviceArn': 'string',
            'serviceName': 'string',
            'clusterArn': 'string',
            'loadBalancers': [
                {
                    'targetGroupArn': 'string',
                    'loadBalancerName': 'string',
                    'containerName': 'string',
                    'containerPort': 123
                },
            ],
            'status': 'string',
            'desiredCount': 123,
            'runningCount': 123,
            'pendingCount': 123,
            'taskDefinition': 'string',
            'deploymentConfiguration': {
                'maximumPercent': 123,
                'minimumHealthyPercent': 123
            },
            'deployments': [
                {
                    'id': 'string',
                    'status': 'string',
                    'taskDefinition': 'string',
                    'desiredCount': 123,
                    'pendingCount': 123,
                    'runningCount': 123,
                    'createdAt': datetime(2015, 1, 1),
                    'updatedAt': datetime(2015, 1, 1)
                },
            ],
            'roleArn': 'string',
            'events': [
                {
                    'id': 'string',
                    'createdAt': datetime(2015, 1, 1),
                    'message': 'string'
                },
            ],
            'createdAt': datetime(2015, 1, 1),
            'placementConstraints': [
                {
                    'type': 'distinctInstance'|'memberOf',
                    'expression': 'string'
                },
            ],
            'placementStrategy': [
                {
                    'type': 'random'|'spread'|'binpack',
                    'field': 'string'
                },
            ]
        }
    }
    
    
    :returns: 
    cluster (string) -- The short name or full Amazon Resource Name (ARN) of the cluster on which to run your service. If you do not specify a cluster, the default cluster is assumed.
    serviceName (string) -- [REQUIRED]
    The name of your service. Up to 255 letters (uppercase and lowercase), numbers, hyphens, and underscores are allowed. Service names must be unique within a cluster, but you can have similarly named services in multiple clusters within a region or across multiple regions.
    
    taskDefinition (string) -- [REQUIRED]
    The family and revision (family:revision ) or full Amazon Resource Name (ARN) of the task definition to run in your service. If a revision is not specified, the latest ACTIVE revision is used.
    
    loadBalancers (list) -- A load balancer object representing the load balancer to use with your service. Currently, you are limited to one load balancer or target group per service. After you create a service, the load balancer name or target group ARN, container name, and container port specified in the service definition are immutable.
    For Elastic Load Balancing Classic load balancers, this object must contain the load balancer name, the container name (as it appears in a container definition), and the container port to access from the load balancer. When a task from this service is placed on a container instance, the container instance is registered with the load balancer specified here.
    For Elastic Load Balancing Application load balancers, this object must contain the load balancer target group ARN, the container name (as it appears in a container definition), and the container port to access from the load balancer. When a task from this service is placed on a container instance, the container instance and port combination is registered as a target in the target group specified here.
    
    (dict) --Details on a load balancer that is used with a service.
    
    targetGroupArn (string) --The full Amazon Resource Name (ARN) of the Elastic Load Balancing target group associated with a service.
    
    loadBalancerName (string) --The name of a Classic load balancer.
    
    containerName (string) --The name of the container (as it appears in a container definition) to associate with the load balancer.
    
    containerPort (integer) --The port on the container to associate with the load balancer. This port must correspond to a containerPort in the service's task definition. Your container instances must allow ingress traffic on the hostPort of the port mapping.
    
    
    
    
    
    desiredCount (integer) -- [REQUIRED]
    The number of instantiations of the specified task definition to place and keep running on your cluster.
    
    clientToken (string) -- Unique, case-sensitive identifier you provide to ensure the idempotency of the request. Up to 32 ASCII characters are allowed.
    role (string) -- The name or full Amazon Resource Name (ARN) of the IAM role that allows Amazon ECS to make calls to your load balancer on your behalf. This parameter is required if you are using a load balancer with your service. If you specify the role parameter, you must also specify a load balancer object with the loadBalancers parameter.
    If your specified role has a path other than / , then you must either specify the full role ARN (this is recommended) or prefix the role name with the path. For example, if a role with the name bar has a path of /foo/ then you would specify /foo/bar as the role name. For more information, see Friendly Names and Paths in the IAM User Guide .
    
    deploymentConfiguration (dict) -- Optional deployment parameters that control how many tasks run during the deployment and the ordering of stopping and starting tasks.
    
    maximumPercent (integer) --The upper limit (as a percentage of the service's desiredCount ) of the number of tasks that are allowed in the RUNNING or PENDING state in a service during a deployment. The maximum number of tasks during a deployment is the desiredCount multiplied by maximumPercent /100, rounded down to the nearest integer value.
    
    minimumHealthyPercent (integer) --The lower limit (as a percentage of the service's desiredCount ) of the number of running tasks that must remain in the RUNNING state in a service during a deployment. The minimum healthy tasks during a deployment is the desiredCount multiplied by minimumHealthyPercent /100, rounded up to the nearest integer value.
    
    
    
    placementConstraints (list) -- An array of placement constraint objects to use for tasks in your service. You can specify a maximum of 10 constraints per task (this limit includes constraints in the task definition and those specified at run time).
    
    (dict) --An object representing a constraint on task placement. For more information, see Task Placement Constraints in the Amazon EC2 Container Service Developer Guide .
    
    type (string) --The type of constraint. Use distinctInstance to ensure that each task in a particular group is running on a different container instance. Use memberOf to restrict selection to a group of valid candidates. Note that distinctInstance is not supported in task definitions.
    
    expression (string) --A cluster query language expression to apply to the constraint. Note you cannot specify an expression if the constraint type is distinctInstance . For more information, see Cluster Query Language in the Amazon EC2 Container Service Developer Guide .
    
    
    
    
    
    placementStrategy (list) -- The placement strategy objects to use for tasks in your service. You can specify a maximum of 5 strategy rules per service.
    
    (dict) --The task placement strategy for a task or service. For more information, see Task Placement Strategies in the Amazon EC2 Container Service Developer Guide .
    
    type (string) --The type of placement strategy. The random placement strategy randomly places tasks on available candidates. The spread placement strategy spreads placement across available candidates evenly based on the field parameter. The binpack strategy places tasks on available candidates that have the least available amount of the resource that is specified with the field parameter. For example, if you binpack on memory, a task is placed on the instance with the least amount of remaining memory (but still enough to run the task).
    
    field (string) --The field to apply the placement strategy against. For the spread placement strategy, valid values are instanceId (or host , which has the same effect), or any platform or custom attribute that is applied to a container instance, such as attribute:ecs.availability-zone . For the binpack placement strategy, valid values are cpu and memory . For the random placement strategy, this field is not used.
    
    
    
    
    
    
    """
    pass