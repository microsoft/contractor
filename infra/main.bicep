//-----------------------------------------------
// main.bicep  – subscription‑scope entry point
//-----------------------------------------------
targetScope = 'subscription'

/*───────────────────────────
   1. Core deployment inputs
───────────────────────────*/
@minLength(1)
@maxLength(64)
param rgName string

@allowed([
  'eastus', 'eastus2', 'westus', 'westus2', 'westcentralus'
  'northeurope', 'francecentral', 'switzerlandnorth', 'switzerlandwest'
  'uksouth', 'australiaeast', 'eastasia', 'southeastasia'
  'centralindia', 'jioindiawest', 'japanwest', 'koreacentral'
])
param location string

@allowed([ 'dev', 'test', 'prod' ])
param environment string = 'dev'

param keyVaultName string = 'contractor-kv'
param subscriptionId string = subscription().subscriptionId

/* ACR settings */
@minLength(5)
@maxLength(50)
param acrName string = 'contractoracr'
var   acrLoginServer = '${acrName}.azurecr.io'

/*--------- OpenAI ---------*/
param openAiAccountName       string = 'contractor-openai'
param openAiCustomDomain      string = 'openai-ctr'
param deployOpenAiAccount     bool   = true
param openAiRestore           bool   = false
param openAiModelDeployment   string = 'contractor-4o'
@allowed([ 'gpt-4o', 'gpt-4o-mini' ])
param openAiModelName         string = 'gpt-4o'
param openAiModelVersion      string = '2024-11-20'
param openAiCapacity          int    = 80

/*--------- Speech ---------*/
param speechAccountName       string = 'contractor-speech'
param speechCustomDomain      string = 'speech-ctr'
param deploySpeechAccount     bool   = true
param speechRestore           bool   = false

/*--------- Vision ---------*/
param visionAccountName       string = 'contractor-vision'
param visionCustomDomain      string = 'vision-ctr'
param deployVisionAccount     bool   = true
param visionRestore           bool   = false

/*--- Document Intelligence ---*/
param docAccountName          string = 'contractor-docint'
param docCustomDomain         string = 'doc-intell-contractor'
param deployDocAccount        bool   = true
param docRestore              bool   = false
param docSku                  string = 'S0'

/*──────────── Resources ────────────*/
resource contractorRG 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: rgName
  location: location
}

resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' existing = {
  scope: contractorRG
  name: keyVaultName
}

/*──────── VNet (returns containerAppsSubnetId) ────────*/
module vnetModule './modules/vnet.bicep' = {
  name: 'deployVnet'
  scope: contractorRG
  params: {
    location: location
    envType: environment
    keyVaultReference: keyVault.name
  }
}

/*──────── Log Analytics ────────*/
module logAnalyticsModule './modules/loga.bicep' = {
  name: 'deployLogAnalytics'
  scope: contractorRG
  params: {
    workspaceName: 'contractor-loganalytics'
    location: location
    retentionInDays: 30
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

/*──────── ACR ────────*/
module acrModule './modules/acr.bicep' = {
  name: 'deployContainerRegistry'
  scope: contractorRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
    acrName: acrName
    sku: 'Standard'
  }
}

/*──────── Cosmos DB (unchanged) ────────*/
module cosmosDbModule './modules/cosmos.bicep' = {
  name: 'deployCosmosDb'
  scope: contractorRG
  params: {
    cosmosDbName: 'contractor-cosmosdb'
    location: location
    resourceGroupLocation: contractorRG.location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

/*──────── Container Apps env + first app ────────*/
module containerAppsEnvModule './modules/aca.bicep' = {
  name: 'deployContainerAppsEnv'
  scope: contractorRG
  params: {
    location: location
    infrastructureSubnetId: vnetModule.outputs.containerAppsSubnetId
    logAnalyticsWorkspaceName: logAnalyticsModule.outputs.workspaceName
    envType: environment
    containerAppName: 'contractor-api'
    containerAppImage: '${acrLoginServer}/api:latest'
    acrLoginServer: acrLoginServer
  }
}

/*──────── Azure OpenAI ────────*/
module openAiModule './modules/aoai.bicep' = {
  name: 'deployAzureOpenAI'
  scope: contractorRG
  params: {
    aiServiceAccountName: openAiAccountName
    deployAccount: deployOpenAiAccount
    restore: openAiRestore
    location: location
    customSubDomainName: openAiCustomDomain
    subnetId: vnetModule.outputs.containerAppsSubnetId
    modelDeploymentName: openAiModelDeployment
    modelName: openAiModelName
    modelVersion: openAiModelVersion
    capacity: openAiCapacity
    tags: { project: 'Contractor', environment: environment }
  }
}

/*──────── Speech Services ────────*/
module speechModule './modules/speech.bicep' = {
  name: 'deploySpeech'
  scope: contractorRG
  params: {
    speechAccountName: speechAccountName
    deployAccount: deploySpeechAccount
    restore: speechRestore
    location: location
    customSubDomainName: speechCustomDomain
    subnetId: vnetModule.outputs.openAiSubnetId
    tags: { project: 'Contractor', environment: environment }
  }
}

/*──────── Vision (Computer Vision) ────────*/
module visionModule './modules/vision.bicep' = {
  name: 'deployVision'
  scope: contractorRG
  params: {
    visionAccountName: visionAccountName
    deployAccount: deployVisionAccount
    restore: visionRestore
    location: location
    customSubDomainName: visionCustomDomain
    subnetId: vnetModule.outputs.openAiSubnetId
    tags: { project: 'Contractor', environment: environment }
  }
}

/*──────── Document Intelligence ────────*/
module documentModule './modules/document.bicep' = {
  name: 'deployDocument'
  scope: contractorRG
  params: {
    docAccountName: docAccountName
    deployAccount: deployDocAccount
    restore: docRestore
    location: location
    sku: docSku
    customSubDomainName: docCustomDomain
    subnetId: vnetModule.outputs.openAiSubnetId
    tags: { project: 'Contractor', environment: environment }
  }
}

// =======================================================================
// Module: Azure Bing Search Resource (applying security best practices)
// =======================================================================
module bingSearch './modules/bing.bicep' = {
  name: 'deployBingSearch'
  scope: contractorRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// =================================================================
// Module: Azure Static Web App (enforces HTTPS and CORS policies)
// =================================================================
module staticWebApp './modules/staticwapp.bicep' = {
  name: 'deployStaticWebApp'
  scope: contractorRG
  params: {
    location: location
    repositoryUrl: 'https://github.com/Azure-Samples/contractor.git'
    environment: environment
    keyVaultReference: keyVault.name
  }
}

// ===================================================================================
// Module: Azure Blob Storage Account (enforces HTTPS-only and uses private endpoints)
// ===================================================================================
module storageAccount './modules/blob.bicep' = {
  name: 'deployStorageAccount'
  scope: contractorRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}
