targetScope = 'resourceGroup'

@description('Location for the Azure Container Apps Environment')
param location string

@description('Name for the Subnet (fully qualified resourceId) to be used for the Container Apps Environment.')
param infrastructureSubnetId string

@description('Keyvault secret for the Log Analytics Workspace.')
param keyVaultReference string

@description('Environment indicator to adjust address space accordingly')
param environment string

@description('Subscription Id for resource references')
param subscriptionId string

resource containerAppsEnv 'Microsoft.App/managedEnvironments@2022-03-01' = {
  name: 'contractor-containerapps-env'
  location: location
  properties: {
    vnetConfiguration: {
      infrastructureSubnetId: infrastructureSubnetId
    }
  }
  tags: {
    project: 'Contractor'
    environment: environment
    keyVaultReference: keyVaultReference
    subscriptionId: subscriptionId
  }
}

output containerAppsEnvId string = containerAppsEnv.id
