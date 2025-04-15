targetScope = 'resourceGroup'

@description('Location for the Azure OpenAI instance')
param location string

@description('Name for the Azure OpenAI instance')
param openAiAccountName string = 'contractor-openai'

@description('Keyvault secret for the Log Analytics Workspace.')
param keyVaultReference string

@description('Environment indicator to adjust address space accordingly')
param environment string

@description('Subscription Id for resource references')
param subscriptionId string = subscription().subscriptionId

resource openAi 'Microsoft.CognitiveServices/accounts@2022-12-01' = {
  name: openAiAccountName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  kind: 'OpenAI'
  properties: {
    apiProperties: {
      subnet: {
        // Updated to use the composite resource type.
        id: resourceId(subscriptionId, resourceGroup().name, 'Microsoft.Network/virtualNetworks/subnets', 'myVNet', 'data')
      }
    }
  }
  tags: {
    project: 'Contractor'
    environment: environment
    keyVaultReference: keyVaultReference
  }
}

output openAiAccountId string = openAi.id
