"""
TURN 服务器配置模块
用于在 Streamlit Cloud 上部署 WebRTC 应用时获取 ICE 服务器配置。
支持 Twilio TURN 服务器和免费的 Google STUN 服务器作为回退方案。
一般情况下STUN服务器可用。
参考: https://github.com/whitphx/streamlit-webrtc
"""
import os
import logging
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data
def get_ice_servers():
    """
    获取 ICE 服务器配置，用于 WebRTC 连接。
    
    优先使用 Twilio TURN 服务器（如果配置了环境变量），
    否则回退到免费的 Google STUN 服务器。
    
    Returns:
        list: ICE 服务器配置列表，格式为 [{"urls": [...]}, ...]
    """
    # 尝试使用 Twilio TURN 服务器
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        # 回退到免费的 STUN 服务器
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        from twilio.rest import Client
        
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        logger.info("Successfully obtained Twilio ICE servers")
        return token.ice_servers
    except Exception as e:
        logger.error(f"Failed to get Twilio ICE servers: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
