
from src.neurotrace.blockchain.fabric_client import fabric_client

try:
    result = fabric_client.health()
    print("Connexion Fabric Gateway réussie !")
    print(result)

    artifact = fabric_client.get_artifact(
        "ARTIFACT_TAMPER_TEST_001"
    )

    print("\nArtefact récupéré depuis Fabric :")
    print(artifact)

    verification = fabric_client.verify_hash(
        "ARTIFACT_TAMPER_TEST_001",
        "A629AADD24C31DAA08D9E79DCE6C1DFCFA77AC95775B4683EC29663C61CBF9DD"
    )

    print("\nRésultat de vérification :")
    print(verification)

except Exception as e:
    print("Erreur :", e)
