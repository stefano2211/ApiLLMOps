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
        "username": "mlops_tester",
        "email": "admin@mlops.test",
        "password": "Password123!"
    }
    reg_resp = requests.post(f"{EDGE_URL}/auth/register", json=user_payload)
    if reg_resp.status_code not in [200, 400]:  # 400 = ya existe, está bien
        print(f"⚠️  Register: {reg_resp.status_code} - {reg_resp.text}")
    
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
        "base_model": "unsloth/Qwen3.5-2B",
        "epochs": 1,
        "webhook_url": "http://host.docker.internal:8000/mlops/webhook/model-ready"
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
    headers = {"x-api-key": MOTHERSHIP_API_KEY}
    payload = {"model_tag": "aura_tenant_01-v2"}
    
    print(f"Llamando Webhook OTA a {url} ...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code in [200, 202]:
            print("✅ Webhook OTA aceptado. Revisa los logs de IndustrialBackend para ver el proceso 'ollama create'.")
        else:
            print(f"❌ Respuesta inesperada: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    return True

def test_4_webhook_without_auth():
    print_header("Fase 4: Verificación de Seguridad — Webhook sin API Key")
    url = f"{EDGE_URL}/mlops/webhook/model-ready"
    # Sin API key -> debe retornar 401 o 422 (campo requerido)
    try:
        resp = requests.post(url, json={"model_tag": "aura_tenant_01-v2"}, timeout=5)
        if resp.status_code in [401, 422]:
            print(f"\u2705 Webhook correctamente rechazado sin API key (HTTP {resp.status_code}).")
        else:
            print(f"\u274c FALLA DE SEGURIDAD: Se esperaba 401/422, se recibió {resp.status_code}")
    except Exception as e:
        print(f"\u274c Error: {e}")

def test_5_webhook_injection():
    print_header("Fase 5: Verificación de Seguridad — Sanitización de model_tag")
    url = f"{EDGE_URL}/mlops/webhook/model-ready"
    headers = {"x-api-key": MOTHERSHIP_API_KEY}
    try:
        resp = requests.post(url, headers=headers, json={"model_tag": "valid-model; rm -rf /"}, timeout=5)
        if resp.status_code == 400:
            print("\u2705 Inyección de shell correctamente bloqueada (HTTP 400).")
        else:
            print(f"\u274c POSIBLE VULNERABILIDAD: Se esperaba 400, se recibió {resp.status_code}")
    except Exception as e:
        print(f"\u274c Error: {e}")

if __name__ == "__main__":
    print("\n--- INICIANDO TEST END-TO-END DE MLOPS ---")
    
    token = get_edge_token()
    if token:
        test_1_trigger_curator(token)
        time.sleep(2)
    
    test_2_trigger_mothership_training()
    time.sleep(2)
    test_3_simulate_webhook()
    time.sleep(1)
    test_4_webhook_without_auth()
    time.sleep(1)
    test_5_webhook_injection()
    
    print("\n--- TEST FINALIZADO ---")
