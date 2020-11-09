import sys

LMA_PATH = "/srv/LocalManagementApp"
sys.path.append(LMA_PATH)

HOST = "127.0.0.1"

# 该字段控制各个模式下框线是否回显
draw_dict = {
    "DEBUG": {
        "DRAW_TRACK": True,
        "DRAW_HEAD": True,
        "DRAW_FACE": True,
        "DRAW_AGE_GENDER": False,
        "DRAW_KEY_POINT": False,
        "DRAW_POSEBOX": False,
        "DRAW_COUNT": True
    },
    "AI": {
        "DRAW_TRACK": False,
        "DRAW_HEAD": True,
        "DRAW_FACE": True,
        "DRAW_AGE_GENDER": True,
        "DRAW_KEY_POINT": True,
        "DRAW_POSEBOX": True,
        "DRAW_COUNT": False
    },
    "REC": {
        "DRAW_TRACK": False,
        "DRAW_HEAD": False,
        "DRAW_FACE": False,
        "DRAW_AGE_GENDER": False,
        "DRAW_KEY_POINT": False,
        "DRAW_POSEBOX": False,
        "DRAW_COUNT": False
    }
}

default_business_params = {
    "ASSESS": {
        "MIN_SIZE": 64,
        "MID_SIZE": 80,
        "MAX_SIZE": 1080,
        "MIN_BOX_SCORE": 0,
        "MAX_ANGLE_YAW": 30,
        "MAX_ANGLE_PITCH": 40,
        "MAX_ANGLE_ROLL": 50,
        "MIN_BRIGHTNESS": 0,
        "MAX_BRIGHTNESS": 255,
        "L2_NORM": 0.3
    },
    "COUNT": {
        "ANGLE": 85,
        "ENTRANCE_DIRECTION": None,
        "ENTRANCE_LINE": None,
        "ROI_AREA": None,
        "VECTOR_LEN": 0.5
    },
    "OTHER": {
        "DISPLAY_MODE": "DEBUG",
        "KPS_ON": True,
        "USE_BRIGHTNESS_ENHANCEMENT": True,
        "BRIGHTNESS_GAIN": 1.5
    },
    "TRACK": {
        "MAX_MISMATCH_TIMES": 15
    },
    "DRAW": {
        "MAX_COLOR_NUM": 79,
        "DRAW_TRACK": True,
        "TRACK_CIRCLE_RADIUS": 1,
        "TRACK_CIRCLE_SIZE": 0,
        "TRACK_LINE_SIZE": 5,
        "TRACK_DRAW_LESS": True,
        "DRAW_TRACK_NUM": 25,
        "DRAW_HEAD": True,
        "HEAD_RECTANGLE_SIZE": 4,
        "DRAW_FACE": True,
        "FACE_RECTANGLE_SIZE": 2,
        "DRAW_AGE_GENDER": False,
        "AGE_GENDER_FONT_SIZE": 20,
        "AGE_GENDER_COLOR": (218, 227, 218),
        "AGE_GENDER_FONT": {"Linux": "NotoSansCJK-Black.ttc", "Windows": "simhei.ttf"},
        "DRAW_KEY_POINT": False,
        "KEY_POINT_CIRCLE_RADIUS": 1,
        "KEY_POINT_CIRCLE_SIZE": -1,
        "KEY_POINT_COLOR": (255, 255, 0),
        "KEY_POINT_LINE_SIZE": 1,
        "DRAW_POSEBOX": False,
        "POSEBOX_POLYLINES_SIZE": 1,
        "POSEBOX_LINE_SIZE": 1,
        "DRAW_COUNT": True,
        "COUNT_IN_ORG": (20, 50),
        "COUNT_OUT_ORG": (20, 100),
        "COUNT_IN_COLOR": (0, 0, 255),
        "COUNT_OUT_COLOR": (0, 255, 0),
        "COUNT_TEXT_FONT_SCALE": 2.0,
        "COUNT_TEXT_SIZE": 5},
    "DATA": {
        "SAVE_LOCAL_PIC": True,
        "LOCAL_PATH": "/srv/Data/picsData/",
        "UPLOAD_BGPIC": False,
        "BGPIC_URL": "http://{}:5000/api/bgpic".format(HOST),
        "BGPIC_INTERVAL": 3600,
        "UPLOAD_HEADPIC": True,
        "HEADPIC_URL": "http://{}:5000/api/headpic".format(HOST),
        "UPLOAD_COUNT": True,
        "COUNT_URL": "http://{}:5000/api/pvcount".format(HOST),
        "UPLOAD_HEARTBEAT": True,
        "HEARTBEAT_URL": "http://{}:5000/api/heartbeat".format(HOST),
        "HEARTBEAT_INTERVAL": 60
    },
    "MONGO": {
        "DB_HOST": HOST,
        "DB_PORT": 27017,
        "DB_NAME": "app"
    },
    "IMAGE": {
        "SHAPE": [720, 1280]
    },
    "DEAL": {
        "IN_AREA": None,
        "OUT_AREA": None
    },
    "INFO": {
        "DEVICE_MAC": None
    }
}
