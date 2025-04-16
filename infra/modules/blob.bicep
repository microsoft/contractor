targetScope = 'resourceGroup'

@description('The name of the Storage Account to be created for Blob Storage.')
param storageAccountName string = 'contractorstorageacct'

@description('The location where the Storage Account will be deployed.')
param location string

@description('The SKU of the Storage Account.')
param storageSku string = 'Standard_LRS'

@description('The kind of Storage Account.')
param storageKind string = 'StorageV2'

@description('The access tier for the Storage Account.')
param accessTier string = 'Hot'

@description('Keyvault secret for the Log Analytics Workspace.')
param keyVaultReference string

@description('Environment indicator to adjust address space accordingly')
param environment string

@description('Subscription Id for resource references')
param subscriptionId string = subscription().subscriptionId

@description('The name of the Virtual Network that contains the subnet for the private endpoint.')
param vnetName string = 'contractor-vnet'

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: storageSku
  }
  kind: storageKind
  properties: {
    accessTier: accessTier
    supportsHttpsTrafficOnly: true
  }
  tags: {
    project: 'Contractor'
    environment: environment
    keyVaultReference: keyVaultReference
    subscriptionId: subscriptionId
  }
}

resource blobPrivateEndpoint 'Microsoft.Network/privateEndpoints@2021-05-01' = {
  name: '${storageAccountName}-pe'
  location: location
  properties: {
    subnet: {
      // Use the fully qualified resourceId for the subnet.
      id: resourceId(subscriptionId, resourceGroup().name, 'Microsoft.Network/virtualNetworks/subnets', vnetName, 'data')
    }
    privateLinkServiceConnections: [
      {
        name: 'blobConnection'
        properties: {
          privateLinkServiceId: storageAccount.id
          groupIds: [
            'blob'
          ]
        }
      }
    ]
  }
}

output storageAccountId string = storageAccount.id
