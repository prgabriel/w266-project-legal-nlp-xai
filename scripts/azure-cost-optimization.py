# Cost optimization strategies
AZURE_DEPLOYMENT_CONFIGS = {
    'development': {
        'sku': 'B1',  # Basic tier
        'min_replicas': 0,  # Scale to zero
        'max_replicas': 1,
        'cpu': '0.5',
        'memory': '1Gi'
    },
    'production': {
        'sku': 'S1',  # Standard tier
        'min_replicas': 1,
        'max_replicas': 5,
        'cpu': '1.0',
        'memory': '2Gi'
    }
}