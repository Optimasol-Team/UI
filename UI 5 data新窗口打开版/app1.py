# app.py
import os, json, csv, io, asyncio, secrets
import sys, socket, subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

import httpx
from fastapi import FastAPI, HTTPException, Depends, Response, Request, Query
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import time


# ===================== Streamlit / Subprocess =====================
DATA_UI_PORT = 8502
_data_ui_proc = None

def _data_ui_entry() -> Path:
    return Path(__file__).parent / "application_v1-main" / "app.py"


STREAMLIT_PORT = 8501
_streamlit_proc = None

def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0

def _streamlit_entry() -> Path:
    return Path(__file__).parent / "Optimiser_Engine-v2.0-main" / "apps" / "simulator_simple" / "main.py"

def _engine_root() -> Path:
    return Path(__file__).parent / "Optimiser_Engine-v2.0-main"


# ===================== Client Config Models =====================
class PositionIn(BaseModel):
    latitude: float
    longitude: float
    altitude: float

class PVConfigIn(BaseModel):
    detail_level: Literal["full", "simple"]
    azimuth: float
    tilt: float
    surface: float

    power_nominal: Optional[float] = None
    inverter_ceiling: Optional[float] = None
    inverter_efficiency: Optional[float] = None
    cable_losses: Optional[float] = None

    global_efficiency: Optional[float] = None
    auto_adjusting_params: Optional[bool] = None

class ClientConfigIn(BaseModel):
    position: PositionIn
    pv_config: PVConfigIn


# ===================== Password hashing + users.json storage =====================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def normalize_password(pw: str) -> str:
    """
    passlib 的某些方案对超长密码可能有处理限制。
    超长时先做 SHA256，再交给 hash。
    """
    b = pw.encode("utf-8")
    if len(b) <= 72:
        return pw
    return hashlib.sha256(b).hexdigest()

USERS_PATH = "users.json"
DEV_ID = "DEVACCOUNT"

def load_users() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def ensure_bootstrap_dev() -> None:
    users = load_users()
    if DEV_ID in users:
        return
    dev_pwd = os.getenv("DEV_PASSWORD")
    if not dev_pwd:
        print("[WARN] DEVACCOUNT not found. Set DEV_PASSWORD ONCE to bootstrap dev account.")
        return

    users[DEV_ID] = {
        "password_hash": pwd_context.hash(normalize_password(dev_pwd)),
        "role": "dev",
        "router_base": None,
    }
    save_users(users)
    print("[OK] Bootstrapped DEVACCOUNT into users.json. You can remove DEV_PASSWORD now.")


# ===================== In-memory stores (demo) =====================
# 简易会话：token -> {router_base, role, exp, server_token, ...}
SESS: Dict[str, Dict[str, Any]] = {}
# 简易设置存储：router_id -> dict
SETTINGS: Dict[str, Dict[str, Any]] = {}
# 简易历史缓存：router_id -> list[dict]
HISTORY: Dict[str, List[Dict[str, Any]]] = {}
# 简易 client 配置存储（位置 + 光伏参数）
CLIENTS: Dict[str, Dict[str, Any]] = {}

# ===================== 可配置区（开发期简化） =====================
ROUTER_MAP = {
    "PVROUTER001": "http://192.168.1.111",
    "DEVBOX": "http://192.168.1.50",
}

ENABLE_REAL_DEVICE = False

MQTT_STATUS = {
    "enabled": False,
    "broker": "mqtt.example.com",
    "port": 1883,
    "connected": False,
    "last_sync": None,
    "publish_hz": 1,
    "sub_topic": "PVROUTER007/DATA",
}


# ===================== FastAPI app =====================
app = FastAPI(title="PV UI Backend", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
ensure_bootstrap_dev()


# ===================== Helpers =====================
def make_token() -> str:
    return secrets.token_hex(16)

def now_ts() -> float:
    return time.time()


# ========== Proxy 到主程序 server.py（真实数据源） ==========
SERVER_BASE = os.getenv("OPTIMASOL_SERVER_BASE", "http://127.0.0.1:8000").rstrip("/")

def _srv_auth_headers(sess: Dict[str, Any]) -> Dict[str, str]:
    token = sess.get("server_token")
    if not token:
        return {}
    return {"Authorization": token}

async def _srv_get(path: str, sess: Dict[str, Any], params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=6) as c:
        r = await c.get(f"{SERVER_BASE}{path}", headers=_srv_auth_headers(sess), params=params)
        r.raise_for_status()
        return r.json()

async def _srv_post(path: str, sess: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=6) as c:
        r = await c.post(f"{SERVER_BASE}{path}", headers=_srv_auth_headers(sess), json=payload)
        r.raise_for_status()
        return r.json()


# ===================== Streamlit endpoints =====================
@app.post("/api/data_ui/start")
async def data_ui_start():
    global _data_ui_proc
    entry = _data_ui_entry()
    if not entry.exists():
        raise HTTPException(500, f"Data UI entry not found: {entry}")

    if _is_port_open(DATA_UI_PORT):
        return {"ok": True, "already_running": True, "url": f"http://127.0.0.1:{DATA_UI_PORT}"}

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(entry),
        "--server.port", str(DATA_UI_PORT),
        "--server.headless", "true",
    ]

    env = os.environ.copy()
    data_root = str(entry.parent)
    env["PYTHONPATH"] = data_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    _data_ui_proc = subprocess.Popen(
        cmd,
        cwd=str(entry.parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t0 = time.time()
    while time.time() - t0 < 4.0:
        if _is_port_open(DATA_UI_PORT):
            return {"ok": True, "already_running": False, "url": f"http://127.0.0.1:{DATA_UI_PORT}"}
        time.sleep(0.1)

    raise HTTPException(500, "Data UI failed to start (port not open).")

@app.get("/api/data_ui/status")
async def data_ui_status():
    running = _is_port_open(DATA_UI_PORT)
    return {"ok": True, "running": running, "url": f"http://127.0.0.1:{DATA_UI_PORT}"}

@app.post("/api/optimiser_ui/start")
async def optimiser_ui_start():
    global _streamlit_proc
    entry = _streamlit_entry()
    if not entry.exists():
        raise HTTPException(500, f"Streamlit entry not found: {entry}")

    if _is_port_open(STREAMLIT_PORT):
        return {"ok": True, "already_running": True, "url": f"http://127.0.0.1:{STREAMLIT_PORT}"}

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(entry),
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
    ]

    env = os.environ.copy()
    root = str(_engine_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    _streamlit_proc = subprocess.Popen(
        cmd,
        cwd=str(_engine_root()),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t0 = time.time()
    while time.time() - t0 < 4.0:
        if _is_port_open(STREAMLIT_PORT):
            return {"ok": True, "already_running": False, "url": f"http://127.0.0.1:{STREAMLIT_PORT}"}
        time.sleep(0.1)

    raise HTTPException(500, "Streamlit failed to start (port not open).")

@app.get("/api/optimiser_ui/status")
async def optimiser_ui_status():
    running = _is_port_open(STREAMLIT_PORT)
    return {"ok": True, "running": running, "url": f"http://127.0.0.1:{STREAMLIT_PORT}"}


# ===================== Auth =====================
class LoginIn(BaseModel):
    router_id: str
    password: str
    lang: Optional[str] = "en"

COOKIE_NAME = "session"
COOKIE_MAX_AGE = 3600 * 8  # 8小时

@app.post("/api/auth/login")
async def login(p: LoginIn, response: Response):
    rid = p.router_id.strip()

    # 1) 优先走主程序 server.py 登录（获取 token）
    server_token: Optional[str] = None
    client_id: Optional[str] = None

    try:
        res = await _srv_post("/api/login", sess={}, payload={"email": rid, "password": p.password})
        server_token = res.get("token")
        client_id = res.get("client_id")
    except Exception:
        server_token = None

    # 2) 如果 server 登录没拿到 token，则回退本地 users.json 登录
    router_base = None
    if not server_token:
        users = load_users()
        u = users.get(rid)
        if not u:
            raise HTTPException(401, "Bad credentials")
        if not pwd_context.verify(normalize_password(p.password), u["password_hash"]):
            raise HTTPException(401, "Bad credentials")

        role = u.get("role", "user")
        router_base = u.get("router_base", None)
    else:
        # server 登录成功：先按 user（你也可以后续通过 /api/me 再细分）
        role = "user"

    token = make_token()
    SESS[token] = {
        "router_id": rid,
        "router_base": router_base,
        "role": role,
        "lang": p.lang or "en",
        "exp": now_ts() + COOKIE_MAX_AGE,
        "server_token": server_token,
        "client_id": client_id,
    }

    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=COOKIE_MAX_AGE,
        path="/",
    )
    return {"ok": True, "router_base": router_base, "role": role, "lang": p.lang or "en"}

def auth(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(401, "Unauthorized")
    sess = SESS.get(token)
    if not sess or sess["exp"] < now_ts():
        raise HTTPException(401, "Session expired")
    return sess

@app.get("/api/auth/me")
async def me(sess=Depends(auth)):
    if sess.get("server_token"):
        try:
            me_data = await _srv_get("/api/me", sess)
            return {
                "router_id": me_data.get("client_id") or sess.get("router_id"),
                "role": sess.get("role", "user"),
                "lang": sess.get("lang", "en"),
                "me_raw": me_data,
            }
        except Exception:
            pass

    return {
        "router_id": sess.get("router_id"),
        "role": sess.get("role", "user"),
        "lang": sess.get("lang", "en"),
    }

@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


# ===================== Device JSON (router) =====================
async def fetch_router_json(base: str) -> Dict[str, Any]:
    if not ENABLE_REAL_DEVICE or not base:
        return {
            "PIN": 420, "PROD": 1836, "TEMP1": 52.1, "SAVED_POWER": 2.7,
            "STATUS_OUT1": True, "STATUS_OUT2": False, "MODEINFO": 13,
            "TOTAL_PROD": 6.8, "INJECT": 1.2,
        }
    async with httpx.AsyncClient(timeout=4) as c:
        r = await c.get(f"{base}/getjson")
        r.raise_for_status()
        return r.json()

async def post_router_cmd(base: str, payload: Dict[str, Any]) -> bool:
    if not ENABLE_REAL_DEVICE or not base:
        await asyncio.sleep(0.2)
        return True
    async with httpx.AsyncClient(timeout=4) as c:
        r = await c.post(f"{base}/rest/api", json=payload)
        return r.status_code == 200


# ===================== P02 Dashboard =====================
@app.get("/api/status")
async def api_status(sess=Depends(auth)):
    if sess.get("server_token"):
        s = await _srv_get("/api/summary", sess)
        temp_latest = (s.get("temperature_latest") or {}).get("temperature")
        prod_meas = s.get("production_measured_latest") or {}
        solar_latest = prod_meas.get("power_w")

        return {
            "grid_w": None,
            "solar_w": solar_latest,
            "temp_c": temp_latest,
            "out1_on": None,
            "out2_on": None,
            "modeinfo": None,
            "predict": {
                "decision_latest": s.get("decision_latest"),
                "production_forecast_latest": s.get("production_forecast_latest"),
            },
            "updated_at": time.time(),
        }

    j = await fetch_router_json(sess.get("router_base"))
    settings = SETTINGS.get(sess["router_id"], {})
    typical = settings.get("typical_shower_time", "07:30")
    next_shower = settings.get("next_shower_time", typical)

    predict = {
        "next_shower": next_shower,
        "expected_temp": j.get("TEMP1", None),
        "solar_enough": bool((j.get("PROD", 0) - j.get("PIN", 0)) > 500),
        "extra_kwh_needed": 0.8,
        "estimated_cost": 0.18,
        "suggested_schedule": ["15:00-16:00 off-peak", "11:00-12:00 solar"],
    }
    return {
        "grid_w": j.get("PIN"), "solar_w": j.get("PROD"),
        "temp_c": j.get("TEMP1"), "saved_kwh_today": j.get("SAVED_POWER"),
        "out1_on": j.get("STATUS_OUT1"), "out2_on": j.get("STATUS_OUT2"),
        "modeinfo": j.get("MODEINFO"), "updated_at": int(now_ts()),
        "predict": predict,
    }


# ===================== P04 Boost =====================
class BoostIn(BaseModel):
    on: bool = True

@app.get("/api/boost")
async def boost_status(sess=Depends(auth)):
    j = await fetch_router_json(sess.get("router_base"))
    active = (j.get("MODEINFO", 0) % 10) == 3
    return {"active": active}

@app.post("/api/boost")
async def boost_on(p: BoostIn, sess=Depends(auth)):
    ok = await post_router_cmd(sess.get("router_base"), {"Out1_mode": 3 if p.on else 0})
    if not ok:
        raise HTTPException(500, "Boost failed")
    return {"ok": True}

@app.delete("/api/boost")
async def boost_off(sess=Depends(auth)):
    ok = await post_router_cmd(sess.get("router_base"), {"Out1_mode": 0})
    if not ok:
        raise HTTPException(500, "Cancel Boost failed")
    return {"ok": True}


# ===================== P05 Settings =====================
class SettingsIn(BaseModel):
    typical_shower_time: Optional[str] = None
    next_shower_time: Optional[str] = None
    desired_temp_next: Optional[float] = None
    guests_expected: Optional[int] = 0
    page_refresh_interval: Optional[int] = 5
    lang: Optional[str] = "en"

@app.get("/api/settings")
async def get_settings(sess=Depends(auth)):
    data = SETTINGS.get(sess["router_id"], {})
    return data or {
        "typical_shower_time": "07:30",
        "next_shower_time": "07:30",
        "desired_temp_next": 50,
        "guests_expected": 0,
        "page_refresh_interval": 5,
        "lang": sess.get("lang", "en"),
    }

@app.post("/api/settings")
async def save_settings(p: SettingsIn, sess=Depends(auth)):
    d = SETTINGS.get(sess["router_id"], {})
    d.update({k: v for k, v in p.dict().items() if v is not None})
    SETTINGS[sess["router_id"]] = d
    return {"ok": True, "settings": d}


# ===================== Client (proxy to server when available) =====================
@app.get("/api/client")
async def get_client(sess=Depends(auth)):
    if sess.get("server_token"):
        return await _srv_get("/api/client", sess)
    cfg = CLIENTS.get(sess["router_id"])
    return cfg or {}

@app.post("/api/client")
async def save_client(p: ClientConfigIn, sess=Depends(auth)):
    if sess.get("server_token"):
        payload = {"client": p.dict()}
        res = await _srv_post("/api/client", sess, payload)
        return {"ok": True, "client": res.get("client", res)}
    CLIENTS[sess["router_id"]] = p.dict()
    return {"ok": True, "client": CLIENTS[sess["router_id"]]}


# ===================== History =====================
@app.get("/api/history")
async def history(metric: str = "TEMP1", range_: str = Query("7d", alias="range"), sess=Depends(auth)):
    if sess.get("server_token"):
        if metric.upper() in ("TEMP1", "TEMPERATURE"):
            n = 7 if range_ == "7d" else 30
            data = await _srv_get("/api/history/temperature", sess, params={"limit": n})
            series = [
                {"t": r["timestamp"], "v": r["temperature"]}
                for r in data.get("records", [])
                if r.get("temperature") is not None
            ]
            return {"metric": metric, "range": range_, "series": series}
        raise HTTPException(400, f"metric '{metric}' not supported by server yet")

    import datetime, random
    n = 7 if range_.endswith("7d") else 30
    base_date = datetime.date.today()
    series = []
    for i in range(n):
        day = (base_date - datetime.timedelta(days=n - 1 - i)).isoformat()
        val = round(random.uniform(0.1, 2.0), 3)
        series.append({"t": day, "v": val})
    return {"metric": metric, "range": range_, "series": series}

@app.get("/api/history/export.csv")
async def export_csv(range_: str = Query("30d", alias="range"), sess=Depends(auth)):
    payload = (await history(range_=range_, sess=sess))["series"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "value"])
    for p in payload:
        w.writerow([p["t"], p["v"]])
    return Response(content=buf.getvalue(), media_type="text/csv")


# ===================== MQTT Status =====================
@app.get("/api/mqtt/status")
async def mqtt_status(sess=Depends(auth)):
    return MQTT_STATUS


# ===================== Dev Debug =====================
class CreateUserIn(BaseModel):
    router_id: str
    password: str
    role: Optional[str] = "user"
    router_base: Optional[str] = None

@app.post("/api/dev/create_user")
async def dev_create_user(p: CreateUserIn, sess=Depends(auth)):
    if sess["role"] != "dev":
        raise HTTPException(403, "Forbidden")

    rid = p.router_id.strip()
    if not rid:
        raise HTTPException(400, "router_id required")
    if rid == DEV_ID:
        raise HTTPException(400, "Cannot overwrite DEVACCOUNT")

    users = load_users()
    if rid in users:
        raise HTTPException(409, "User already exists")

    users[rid] = {
        "password_hash": pwd_context.hash(normalize_password(p.password)),
        "role": p.role or "user",
        "router_base": p.router_base
    }
    save_users(users)
    return {"ok": True, "router_id": rid, "role": users[rid]["role"], "router_base": users[rid]["router_base"]}

@app.get("/api/dev/routers")
async def dev_list(sess=Depends(auth)):
    if sess["role"] != "dev":
        raise HTTPException(403, "Forbidden")
    rows = [{"router_id": k, "router_base": v} for k, v in ROUTER_MAP.items()]
    return {"routers": rows}

class DevCmdIn(BaseModel):
    router_id: str
    cmd: Dict[str, Any]

@app.post("/api/dev/cmd")
async def dev_cmd(p: DevCmdIn, sess=Depends(auth)):
    if sess["role"] != "dev":
        raise HTTPException(403, "Forbidden")
    base = ROUTER_MAP.get(p.router_id)
    if not base:
        raise HTTPException(404, "Router not found")
    ok = await post_router_cmd(base, p.cmd)
    return {"ok": ok}


# ===================== Static =====================
app.mount("/", StaticFiles(directory="static", html=True), name="static")
