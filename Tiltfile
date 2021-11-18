def microservice(service):
    k8s_yaml('kubernetes/local.yml')
    docker_build(service['name'], '.', service.get('docker-args', {}))
    k8s_resource(service['name'], port_forwards='%d' % service['port'])

microservice({'name': 'finance-ml', 'port': 22222})
