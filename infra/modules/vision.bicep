// vision.bicep
// Azure Computer Vision module

targetScope = 'resourceGroup'

@description('Name of the Computer Vision account')
param visionAccountName string

@description('Create or recover the account?')
param deployAccount bool = true

@description('Set true to recover a soft-deleted account')
param restore bool = false

@description('Azure region')
param location string = resourceGroup().location

@description('Pricing tier')
@allowed([ 'S1', 'S0' ])
param sku string = 'S1'

@description('Custom sub-domain prefix')
param customSubDomainName string

@description('Subnet resource ID for private access')
param subnetId string

@description('Tags to apply')
param tags object = {}

resource visionService 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployAccount) {
  name: visionAccountName
  location: location
  kind: 'ComputerVision'
  identity: { type: 'SystemAssigned' }
  sku: { name: sku }
  properties: {
    customSubDomainName: customSubDomainName
    apiProperties: { subnet: { id: subnetId } }
    restore: restore
    publicNetworkAccess: 'Disabled'
    networkAcls: { defaultAction: 'Deny', virtualNetworkRules: [ { id: subnetId, ignoreMissingVnetServiceEndpoint: false } ] }
  }
  tags: tags
}

resource visionExisting 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = if (!deployAccount) { name: visionAccountName }

output visionAccountId string = deployAccount ? visionService.id : visionExisting.id
output visionEndpoint   string = deployAccount ? visionService.properties.endpoint : visionExisting.properties.endpoint
