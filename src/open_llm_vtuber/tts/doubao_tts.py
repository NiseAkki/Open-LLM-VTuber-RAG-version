import base64
import json
import uuid
import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """
    豆包语音合成TTS引擎实现
    """

    def __init__(
        self,
        appid: str = "3347038862",
        access_token: str = "l6R1ImK-2zPlwI7U34eu4MsuaAUrfpJp",
        cluster: str = "volcano_tts",
        voice_type: str = "BV051_streaming",
        host: str = "openspeech.bytedance.com",
    ):
        """
        初始化豆包TTS引擎

        Args:
            appid: 平台申请的appid
            access_token: 平台申请的access token
            cluster: 集群名称,默认为"volcano_tts"
            voice_type: 音色类型,默认为"BV051_streaming"
            host: API主机地址,默认为"openspeech.bytedance.com"
        """
        self.appid = appid
        self.access_token = access_token
        self.cluster = cluster
        self.voice_type = voice_type
        self.host = host
        
        self.api_url = f"https://{host}/api/v1/tts"
        self.headers = {"Authorization": f"Bearer;{access_token}"}
        
        # 设置音频文件相关参数
        self.file_extension = "mp3"
        self.new_audio_dir = "cache"

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        生成语音文件

        Args:
            text: 要转换的文本
            file_name_no_ext: 输出文件名(不含扩展名)

        Returns:
            str: 生成的音频文件路径
        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        # 构建请求数据
        request_data = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": self.cluster
            },
            "user": {
                "uid": "388808087185088"
            },
            "audio": {
                "voice_type": self.voice_type,
                "encoding": "mp3",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"
            }
        }

        try:
            # 发送请求
            response = requests.post(
                self.api_url,
                data=json.dumps(request_data),
                headers=self.headers
            )
            
            # 检查响应
            resp_json = response.json()
            if "data" not in resp_json:
                logger.error(f"豆包TTS API返回错误: {resp_json}")
                return None
                
            # 解码音频数据并保存
            audio_data = base64.b64decode(resp_json["data"])
            with open(file_name, "wb") as f:
                f.write(audio_data)
                
            return file_name

        except Exception as e:
            logger.error(f"豆包TTS生成音频失败: {str(e)}")
            return None 