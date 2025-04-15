targetScope = 'resourceGroup'

@description('Name of the virtual network')
param vnetName string = 'contractor-vnet'

@description('The location for the virtual network.')
param location string

@description('Keyvault secret for the Log Analytics Workspace.')
param keyVaultReference string

@description('Environment indicator to adjust address space accordingly')
param environment string

@description('Subscription Id for resource references')
param subscriptionId string = subscription().subscriptionId

// --------------------------------------------------------------------
// Define the NSG for management subnet with an SSH rule.
// Note: In production, consider replacing wildcard address prefixes
// with specific IP ranges to restrict access.
// --------------------------------------------------------------------
resource managementNSG 'Microsoft.Network/networkSecurityGroups@2020-11-01' = {
  name: 'mgmt-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'Allow-SSH'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '22'
          sourceAddressPrefix: '*' // TODO: Replace with trusted IP ranges.
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 1000
          direction: 'Inbound'
        }
      }
    ]
  }
  tags: {
    environment: environment
    keyVaultReference: keyVaultReference
  }
}

// --------------------------------------------------------------------
// Define the NSG for the data subnet with a rule for HTTPS traffic.
// Note: Consider restricting the sourceAddressPrefix as needed.
// --------------------------------------------------------------------
resource dataNSG 'Microsoft.Network/networkSecurityGroups@2020-11-01' = {
  name: 'data-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'Allow-AppTraffic'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: '*' // TODO: Restrict to expected traffic sources.
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 1000
          direction: 'Inbound'
        }
      }
    ]
  }
  tags: {
    environment: environment
    keyVaultReference: keyVaultReference
  }
}

// --------------------------------------------------------------------
// Create the Virtual Network with subnets associated with the NSGs.
// We use a /22 address space for the VNet. This accommodates:
// - A management subnet of /23 (minimum required for Container Apps)
// - And another subnet ("data") within the same VNet.
// --------------------------------------------------------------------
resource virtualNetwork 'Microsoft.Network/virtualNetworks@2021-02-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/22'
      ]
    }
    subnets: [
      {
        name: 'management'
        properties: {
          addressPrefix: '10.0.0.0/23'
          networkSecurityGroup: {
            id: managementNSG.id
          }
        }
      }
      {
        name: 'data'
        properties: {
          addressPrefix: '10.0.2.0/24'
          networkSecurityGroup: {
            id: dataNSG.id
          }
        }
      }
    ]
  }
  tags: {
    environment: environment
    keyVaultReference: keyVaultReference
  }
}

// Output the fully qualified resource ID of the management subnet for Container Apps.
output containerAppsSubnetId string = resourceId(subscriptionId, resourceGroup().name, 'Microsoft.Network/virtualNetworks/subnets', vnetName, 'management')
