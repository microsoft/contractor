targetScope = 'resourceGroup'

@description('The location for the Bing Search resource.')
param location string

@description('The name of the Bing Search resource.')
param bingSearchName string = 'contractor-bingsearch'

@description('Keyvault secret for the Log Analytics Workspace.')
param keyVaultReference string

@description('Environment indicator to adjust address space accordingly')
param environment string

@description('Subscription Id for resource references')
param subscriptionId string = subscription().subscriptionId

resource bingSearch 'Microsoft.Search/searchservices@2020-08-01' = {
  name: bingSearchName
  location: location
  sku: {
    name: 'standard'
  }
  properties: {
    // Optionally, if a newer API version or your design permits,
    // consider applying network restrictions or private endpoint configurations
    // to limit public access.
  }
  tags: {
    project: 'Contractor'
    environment: environment
    keyVaultReference: keyVaultReference
    subscriptionId: subscriptionId
  }
}

output bingSearchId string = bingSearch.id
