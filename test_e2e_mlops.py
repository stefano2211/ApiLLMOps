import requests
import time
import json
import sys

# Configuraciones de Test
EDGE_URL = "http://localhost:8000"
MOTHERSHIP_URL = "http://localhost:8001"
MOTHERSHIP_API_KEY = "default-mothership-secret-key"

def print_header(title):
    print(f"\n{'='*50}\n[TEST] {title}\n{'='*50}")

def get_edge_token():
    print_header("Fase 0: Obteniendo Access Token de IndustrialBackend")
    user_payload = {
        "email": "admin@mlops.test",
        "password": "Password123!",
        "full_name": "MLOps Tester"
    }
    requests.post(f"{EDGE_URL}/auth/register", json=user_payload)
    
    login_payload = {
        "email": "admin@mlops.test",
        "password": "Password123!"
    }
    resp = requests.post(f"{EDGE_URL}/auth/login", json=login_payload)
    if resp.status_code == 200:
        token = resp.json().get("access_token")
        print("✅ Token obtenido exitosamente.")
        return token
    else:
        print(f"❌ Falló el login: {resp.status_code} - {resp.text}")
        return None

def test_1_trigger_curator(token):
    print_header("Fase 1: Disparando Curador Edge (IndustrialBackend)")
    url = f"{EDGE_URL}/curator/run-daily"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"Llamando a {url} ...")
    try:
        response = requests.post(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Respuesta: {response.json()}")
    except Exception as e:
        print(f"❌ Error conectando a IndustrialBackend: {e}")
        return False
    return True

def test_2_trigger_mothership_training():
    print_header("Fase 2: Simular Disparo de Entrenamiento en Nube (ApiLLMOps)")
    url = f"{MOTHERSHIP_URL}/api/v1/training/job"
    headers = {"x-api-key": MOTHERSHIP_API_KEY}
    payload = {
        "tenant_id": "aura_tenant_01",
        "base_model": "unsloth/Qwen2.5-3B-bnb-4bit",
        "epochs": 1,
        "webhook_url": f"{EDGE_URL}/mlops/webhook/model-ready"
    }
    
    print(f"Llamando a {url} ...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Respuesta: {response.json()}")
        if response.status_code == 200:
            print("✅ Se ha encolado el Job en Celery (Revisa los logs de Celery de ApiLLMOps)")
    except Exception as e:
        print(f"❌ Error conectando a ApiLLMOps: {e}")
        return False
    return True

def test_3_simulate_webhook():
    print_header("Fase 3: Simulando Webhook (Mothership -> Edge)")
    url = f"{EDGE_URL}/mlops/webhook/model-ready"
    payload = {"model_tag": "aura_tenant_01-v2"}
    
    print(f"Llamando Webhook OTA a {url} ...")
    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"Status: {response.status_code}")
        print("✅ Si el código es 200/202, revisa los logs de IndustrialBackend: Debería estar intentando hacer 'ollama pull aura_tenant_01-v2'.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    return True

if __name__ == "__main__":
    print("\n--- INICIANDO TEST END-TO-END DE MLOPS ---")
    
    token = get_edge_token()
    if token:
        test_1_trigger_curator(token)
        time.sleep(2)
    
    test_2_trigger_mothership_training()
    time.sleep(2)
    test_3_simulate_webhook()
    
    print("\n--- TEST FINALIZADO ---")
