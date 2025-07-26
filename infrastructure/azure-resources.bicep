param location string = resourceGroup().location
param appName string = 'legal-nlp-toolkit'

// Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: 'acr${appName}${uniqueString(resourceGroup().id)}'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
  }
}

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Container App with placeholder image
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    environmentId: containerAppsEnvironment.id
    configuration: {
      secrets: [
        {
          name: 'registry-password'
          value: containerRegistry.listCredentials().passwords[0].value
        }
      ]
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.listCredentials().username
          passwordSecretRef: 'registry-password'
        }
      ]
      ingress: {
        external: true
        targetPort: 80  // Changed from 8501 to 80 for placeholder image
      }
    }
    template: {
      containers: [
        {
          name: 'placeholder-app'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'  // Placeholder image
          resources: {
            cpu: json('0.25')  // Reduced resources for placeholder
            memory: '0.5Gi'
          }
          env: [
            {
              name: 'PLACEHOLDER_MESSAGE'
              value: 'Legal NLP Toolkit - Building and deploying...'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1  // Single replica for placeholder
      }
    }
  }
}

// Output important values for next steps
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output containerRegistryName string = containerRegistry.name
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output resourceGroupName string = resourceGroup().name