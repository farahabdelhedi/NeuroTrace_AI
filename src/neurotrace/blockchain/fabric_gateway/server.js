
'use strict';

const express = require('express');
const grpc = require('@grpc/grpc-js');
const fs = require('fs');
const path = require('path');

const {
    connect,
    signers
} = require('@hyperledger/fabric-gateway');

const app = express();
app.use(express.json({ limit: '1mb' }));

// Configuration Hyperledger Fabric
const TEST_NETWORK = '/home/farah/hyperledger/fabric-samples/test-network';

const ORG1 = path.join(
    TEST_NETWORK,
    'organizations/peerOrganizations/org1.example.com'
);

const CERT_PATH = path.join(
    ORG1,
    'users/User1@org1.example.com/msp/signcerts/cert.pem'
);

const KEYSTORE_PATH = path.join(
    ORG1,
    'users/User1@org1.example.com/msp/keystore'
);

const TLS_CERT_PATH = path.join(
    ORG1,
    'peers/peer0.org1.example.com/tls/ca.crt'
);

const MSP_ID = 'Org1MSP';
const PEER_ENDPOINT = 'localhost:7051';
const PEER_HOST_ALIAS = 'peer0.org1.example.com';

const CHANNEL_NAME = 'neurotracechannel';
const CHAINCODE_NAME = 'neurotrace';
const CONTRACT_NAME = 'NeuroTraceContract';

let gateway;
let grpcClient;
let contract;

// Connexion à Hyperledger Fabric
function connectToFabric() {
    console.log('Connexion à Hyperledger Fabric...');

    const credentials = fs.readFileSync(CERT_PATH);

    const privateKeyFiles = fs.readdirSync(KEYSTORE_PATH);

    if (privateKeyFiles.length === 0) {
        throw new Error('Aucune clé privée trouvée dans le keystore.');
    }

    const privateKeyPath = path.join(
        KEYSTORE_PATH,
        privateKeyFiles[0]
    );

    const privateKey = require('crypto').createPrivateKey(
        fs.readFileSync(privateKeyPath)
    );

    const tlsRootCert = fs.readFileSync(TLS_CERT_PATH);

    grpcClient = new grpc.Client(
        PEER_ENDPOINT,
        grpc.credentials.createSsl(tlsRootCert),
        {
            'grpc.ssl_target_name_override': PEER_HOST_ALIAS,
            'grpc.default_authority': PEER_HOST_ALIAS
        }
    );

    gateway = connect({
        client: grpcClient,
        identity: {
            mspId: MSP_ID,
            credentials
        },
        signer: signers.newPrivateKeySigner(privateKey)
    });

    const network = gateway.getNetwork(CHANNEL_NAME);

    contract = network.getContract(
        CHAINCODE_NAME,
        CONTRACT_NAME
    );

    console.log('Connexion Fabric initialisée.');
    console.log('Channel :', CHANNEL_NAME);
    console.log('Chaincode :', CHAINCODE_NAME);
}

// Vérifier le fonctionnement du serveur
app.get('/health', (req, res) => {
    res.json({
        success: true,
        message: 'Fabric Gateway opérationnel',
        channel: CHANNEL_NAME,
        chaincode: CHAINCODE_NAME
    });
});

// Enregistrer un artefact
app.post('/artifacts', async (req, res) => {
    try {
        const artifact = req.body;

        const result = await contract.submitTransaction(
            'RegisterArtifact',
            JSON.stringify(artifact)
        );

        res.json(JSON.parse(Buffer.from(result).toString()));
    } catch (error) {
        console.error('Erreur RegisterArtifact :', error.message);

        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Récupérer un artefact
app.get('/artifacts/:id', async (req, res) => {
    try {
        const result = await contract.evaluateTransaction(
            'GetArtifact',
            req.params.id
        );

        res.json(JSON.parse(Buffer.from(result).toString()));
    } catch (error) {
        console.error('Erreur GetArtifact :', error.message);

        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Vérifier le hash SHA-256
app.post('/artifacts/:id/verify', async (req, res) => {
    try {
        const { sha256 } = req.body;

        if (!sha256) {
            return res.status(400).json({
                success: false,
                error: 'Le champ sha256 est obligatoire.'
            });
        }

        const result = await contract.evaluateTransaction(
            'VerifyHash',
            req.params.id,
            sha256
        );

        res.json(JSON.parse(Buffer.from(result).toString()));
    } catch (error) {
        console.error('Erreur VerifyHash :', error.message);

        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Consulter l'historique d'un artefact
app.get('/artifacts/:id/history', async (req, res) => {
    try {
        const result = await contract.evaluateTransaction(
            'GetArtifactHistory',
            req.params.id
        );

        res.json(JSON.parse(Buffer.from(result).toString()));
    } catch (error) {
        console.error('Erreur GetArtifactHistory :', error.message);

        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Démarrage du serveur
const PORT = 3001;

try {
    connectToFabric();

    app.listen(PORT, '0.0.0.0', () => {
        console.log(`Fabric Gateway disponible sur http://localhost:${PORT}`);
    });
} catch (error) {
    console.error('Impossible de démarrer Fabric Gateway :', error);
    process.exit(1);
}

// Fermeture propre
function shutdown() {
    console.log('Fermeture de Fabric Gateway...');

    if (gateway) gateway.close();
    if (grpcClient) grpcClient.close();

    process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
