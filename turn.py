"""
TURN 服务器配置模块

用于在 Streamlit Cloud 上部署 WebRTC 应用时获取 ICE 服务器配置。
支持 Twilio TURN 服务器和免费的 Google STUN 服务器作为回退方案。
"""
import os
import logging

logger = logging.getLogger(__name__)

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    Client = None


def get_ice_servers():
    """
    获取 ICE 服务器配置，用于 WebRTC 连接。
    
    优先使用 Twilio TURN 服务器（如果配置了环境变量），
    否则回退到免费的 Google STUN 服务器。
    
    Returns:
        list: ICE 服务器配置列表，格式为 [{"urls": [...]}, ...]
    """
    # 尝试使用 Twilio TURN 服务器
    if TWILIO_AVAILABLE:
        try:
            account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
            
            if account_sid and auth_token:
                client = Client(account_sid, auth_token)
                token = client.tokens.create()
                logger.info("✅ 使用 Twilio TURN 服务器")
                return token.ice_servers
        except Exception as e:
            logger.warning(f"⚠️ Twilio 配置失败，回退到免费 STUN 服务器: {e}")
    
    # 回退到免费的 Google STUN 服务器
    logger.info("ℹ️ 使用免费的 Google STUN 服务器")
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

