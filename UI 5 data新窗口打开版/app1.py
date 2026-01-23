# app.py
import os, time, json, csv, io, asyncio, secrets
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, Depends, Response, Request
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import sys, time, socket, subprocess
from pathlib import Path
from fastapi import HTTPException

DATA_UI_PORT = 8502
_data_ui_proc = None

def _data_ui_entry() -> Path:
     return Path(r"E:\86181\Desktop\projet\UI\UI 5 data新窗口打开版\application_v1-main\app.py")

#======加client=====
from typing import Literal  # 文件开头的 import 行里要补上 Literal

class PositionIn(BaseModel):
    latitude: float
    longitude: float
    altitude: float

class PVConfigIn(BaseModel):
    # detail_level: "full" = 用户知道全部技术细节; "simple" = 只填少量信息+整体效率
    detail_level: Literal["full", "simple"]

    # 公共字段
    azimuth: float
    tilt: float
    surface: float

    # detail_level = "full" 时使用：
    power_nominal: Optional[float] = None
    inverter_ceiling: Optional[float] = None
    inverter_efficiency: Optional[float] = None
    cable_losses: Optional[float] = None

    # detail_level = "simple" 时使用：
    global_efficiency: Optional[float] = None
    auto_adjusting_params: Optional[bool] = None

class ClientConfigIn(BaseModel):
    position: PositionIn
    pv_config: PVConfigIn

# ===== Password hashing + users.json storage =====
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
import hashlib

def normalize_password(pw: str) -> str:
    """
    bcrypt 有 72 bytes 限制。
    超过 72 bytes 就先做 SHA256（不可逆），再交给 bcrypt。
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
    """
    第一次运行时：如果 users.json 不存在 DEVACCOUNT，
    就从环境变量 DEV_PASSWORD 读取一次明文密码，生成 hash 写入 users.json。
    写完以后你就再也不需要 DEV_PASSWORD 了。
    """
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
        "router_base": None
    }
    save_users(users)
    print("[OK] Bootstrapped DEVACCOUNT into users.json. You can remove DEV_PASSWORD now.")

    # 简易设置存储：router_id -> dict
SETTINGS: Dict[str, Dict[str, Any]] = {}
# 简易历史缓存：router_id -> list[dict]  （真实项目请用SQLite）
HISTORY: Dict[str, List[Dict[str, Any]]] = {}

# 新增：简易 client 配置存储（位置 + 光伏参数）
CLIENTS: Dict[str, Dict[str, Any]] = {}



# ===================== 可配置区（开发期简化） =====================
# 路由器ID -> 实际设备地址（示例）。以后可改成数据库/配置文件。
ROUTER_MAP = {
    "PVROUTER001": "http://192.168.1.111",
    "DEVBOX": "http://192.168.1.50",
}


# 是否允许直接访问真机（没有设备时留 False 用占位数据）
ENABLE_REAL_DEVICE = False

# MQTT（占位：这里仅返回状态，不做真实连接。以后接入 paho-mqtt）
MQTT_STATUS = {
    "enabled": False,
    "broker": "mqtt.example.com",
    "port": 1883,
    "connected": False,
    "last_sync": None,
    "publish_hz": 1,
    "sub_topic": "PVROUTER007/DATA",
}

# ===================== 应用基本设置 =====================
app = FastAPI(title="PV UI Backend", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
ensure_bootstrap_dev()

STREAMLIT_PORT = 8501
_streamlit_proc = None

def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0

def _streamlit_entry() -> Path:
    # ✅ 你已经确认入口在这里
    return Path(__file__).parent / "Optimiser_Engine-v2.0-main" / "apps" / "simulator_simple" / "main.py"

def _engine_root() -> Path:
    return Path(__file__).parent / "Optimiser_Engine-v2.0-main"

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
    _data_ui_proc = subprocess.Popen(
        cmd,
        cwd=str(entry.parent),
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
    """
    Start optimiser Streamlit UI (DEMO/AUDIT purpose).
    Returns the URL to open.
    """
    global _streamlit_proc

    entry = _streamlit_entry()
    if not entry.exists():
        raise HTTPException(500, f"Streamlit entry not found: {entry}")

    # already running
    if _is_port_open(STREAMLIT_PORT):
        return {"ok": True, "already_running": True, "url": f"http://127.0.0.1:{STREAMLIT_PORT}"}

    # start streamlit using current venv python
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(entry),
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
    ]
    _streamlit_proc = subprocess.Popen(
        cmd,
        cwd=str(_engine_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # wait for port to open
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


# 简易会话：token -> {router_base, role, exp}
SESS: Dict[str, Dict[str, Any]] = {}
# 简易设置存储：router_id -> dict
SETTINGS: Dict[str, Dict[str, Any]] = {}
# 简易历史缓存：router_id -> list[dict]  （真实项目请用SQLite）
HISTORY: Dict[str, List[Dict[str, Any]]] = {}

def make_token() -> str:
    return secrets.token_hex(16)

def now_ts() -> float:
    return time.time()

# ===================== P01 登录 =====================
class LoginIn(BaseModel):
    router_id: str
    password: str
    lang: Optional[str] = "en"

COOKIE_NAME = "session"
COOKIE_MAX_AGE = 3600 * 8  # 8小时

@app.post("/api/auth/login")
async def login(p: LoginIn, response: Response):
    rid = p.router_id.strip()

    users = load_users()
    u = users.get(rid)
    if not u:
        raise HTTPException(401, "Bad credentials")

    if not pwd_context.verify(normalize_password(p.password), u["password_hash"]):
        raise HTTPException(401, "Bad credentials")

    role = u.get("role", "user")
    router_base = u.get("router_base", None)

    token = make_token()
    SESS[token] = {
        "router_id": rid,
        "router_base": router_base,
        "role": role,
        "lang": p.lang or "en",
        "exp": now_ts() + COOKIE_MAX_AGE,
    }

    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,   # 上线 HTTPS 后改 True
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

# ===================== 工具：抓取设备JSON =====================
async def fetch_router_json(base: str) -> Dict[str, Any]:
    if not ENABLE_REAL_DEVICE or not base:
        # 占位数据（可快速联调前端）
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

@app.get("/api/auth/me")
async def me(sess=Depends(auth)):
    return {
        "router_id": sess.get("router_id"),
        "role": sess.get("role", "user"),
        "lang": sess.get("lang", "en"),
    }
   
@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


# ===================== P02 仪表盘：状态 + 预测 =====================
@app.get("/api/status")
async def api_status(sess=Depends(auth)):
    j = await fetch_router_json(sess.get("router_base"))
    # 简单预测占位（真实实现请替换为算法）
    settings = SETTINGS.get(sess["router_id"], {})
    typical = settings.get("typical_shower_time", "07:30")
    next_shower = settings.get("next_shower_time", typical)
    desired = settings.get("desired_temp_next", 50)
    predict = {
        "next_shower": next_shower,
        "expected_temp": j.get("TEMP1", None),
        "solar_enough": bool((j.get("PROD", 0) - j.get("PIN", 0)) > 500),
        "extra_kwh_needed": 0.8,
        "estimated_cost": 0.18,  # EUR
        "suggested_schedule": ["15:00-16:00 off-peak", "11:00-12:00 solar"],
    }
    return {
        "grid_w": j.get("PIN"), "solar_w": j.get("PROD"),
        "temp_c": j.get("TEMP1"), "saved_kwh_today": j.get("SAVED_POWER"),
        "out1_on": j.get("STATUS_OUT1"), "out2_on": j.get("STATUS_OUT2"),
        "modeinfo": j.get("MODEINFO"), "updated_at": int(now_ts()),
        "predict": predict,
    }

# ===================== P04 Boost 控制 =====================
class BoostIn(BaseModel):
    on: bool = True

@app.get("/api/boost")
async def boost_status(sess=Depends(auth)):
    # 占位：以 modeinfo 的低位判断
    j = await fetch_router_json(sess.get("router_base"))
    active = (j.get("MODEINFO", 0) % 10) == 3
    return {"active": active}

@app.post("/api/boost")
async def boost_on(p: BoostIn, sess=Depends(auth)):
    ok = await post_router_cmd(sess.get("router_base"), {"Out1_mode": 3 if p.on else 0})
    if not ok: raise HTTPException(500, "Boost failed")
    return {"ok": True}

@app.delete("/api/boost")
async def boost_off(sess=Depends(auth)):
    ok = await post_router_cmd(sess.get("router_base"), {"Out1_mode": 0})
    if not ok: raise HTTPException(500, "Cancel Boost failed")
    return {"ok": True}

# ===================== P05 设置 =====================
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
    # 这里可触发预测模块更新
    return {"ok": True, "settings": d}


# ===================== Client（位置 + 光伏参数） =====================

@app.get("/api/client")
async def get_client(sess=Depends(auth)):
    """
    返回当前 router_id 对应的 client 配置（如果不存在则返回 {}）
    """
    cfg = CLIENTS.get(sess["router_id"])
    return cfg or {}

@app.post("/api/client")
async def save_client(p: ClientConfigIn, sess=Depends(auth)):
    """
    保存/更新当前 router 的 client 配置。
    目前先保存在内存 dict 里；之后可以在这里调用 MeteoManager.
    """
    CLIENTS[sess["router_id"]] = p.dict()

    # TODO （下一步集成 Weather v2.0 的位置）:
    # from MeteoManager import MeteoManager
    # from models import Client, Position, Requetes, Features, Installation_PV
    # manager = MeteoManager(path_bdd=Path("data/meteo_router.sqlite"))
    # -> 把 p 转换为 Position / Installation_PV / Features / Client
    # -> manager.client.create_client(client)

    return {"ok": True, "client": CLIENTS[sess["router_id"]]}


# ===================== P03 历史与导出（占位数据） =====================
from fastapi import Query

@app.get("/api/history")
async def history(
    metric: str = "SAVED_Cost",
    range_: str = Query("7d", alias="range"),
    sess=Depends(auth)
):
    import datetime, random

    n = 7 if range_.endswith("7d") else 30
    base_date = datetime.date.today()

    series = []
    for i in range(n):   # ✅ 这里的 range 是 Python 内置函数了
        day = (base_date - datetime.timedelta(days=n-1-i)).isoformat()
        val = round(random.uniform(0.1, 2.0), 3)
        series.append({"t": day, "v": val})

    return {
        "metric": metric,
        "range": range_,
        "series": series
    }



from fastapi import Query

@app.get("/api/history/export.csv")
async def export_csv(range_: str = Query("30d", alias="range"), sess=Depends(auth)):
    payload = (await history(range_=range_, sess=sess))["series"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "value"])
    for p in payload:
        w.writerow([p["t"], p["v"]])
    return Response(content=buf.getvalue(), media_type="text/csv")

# ===================== P06 MQTT 状态（占位） =====================
@app.get("/api/mqtt/status")
async def mqtt_status(sess=Depends(auth)):
    return MQTT_STATUS

# ===================== P07 开发者调试（可选） =====================
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
    if sess["role"] != "dev": raise HTTPException(403, "Forbidden")
    rows = [{"router_id": k, "router_base": v} for k, v in ROUTER_MAP.items()]
    return {"routers": rows}

class DevCmdIn(BaseModel):
    router_id: str
    cmd: Dict[str, Any]

@app.post("/api/dev/cmd")
async def dev_cmd(p: DevCmdIn, sess=Depends(auth)):
    if sess["role"] != "dev": raise HTTPException(403, "Forbidden")
    base = ROUTER_MAP.get(p.router_id)
    if not base: raise HTTPException(404, "Router not found")
    ok = await post_router_cmd(base, p.cmd)
    return {"ok": ok}

app.mount("/", StaticFiles(directory="static", html=True), name="static")