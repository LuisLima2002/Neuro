from enum import auto, Enum
from pathlib import Path
import time

import numpy as np
from custom.gemini import VoiceIA
from custom.goto_standby import pos_standby
from custom.smol_run import RobotIA
from custom.open import openGripper
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from typing import Any
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
from yolo.yolo import YOLOModel


class_to_request = {
    "yellow": "put aside the yellow screwdriver in the third rectangle, and go to standby position",
    "red": "put aside the red screwdriver in the second rectangle, and go to standby position",
    "blue": "put aside the blue screwdriver in the fourth rectangle, and go to standby position",
}


class RobotStates(Enum):
    STANDBY = auto()
    STORE_OBJECT = auto()
    RETRIEVE_OBJECT = auto()
    GOTO_STORAGE = auto()
    WAIT_VOICE_RELEASE = auto()
    GOTO_STANDBY = auto()


class StateRunner:
    def __init__(self):
        self.running = False

    def run_states(self):
        self._prepare_systems()
        self.set_state(RobotStates.STANDBY)

        self.running = True
        while self.running:
            print(f"Current state: {self.state.name.title()}")
            run_state = getattr(self, self.state.name.lower())
            run_state()
            time.sleep(0.5)
        print("Robot stopped")

    def _prepare_systems(self):
        self.yolo = self._prepare_yolo()

        self.robot = self._prepare_robot()
        self.robot_cam = self.robot.cameras["robo"]
        pos_standby(self.robot)

        self.robotIA= RobotIA(self.robot,"/home/fablab/lerobot/outputs/train/Neurof14K/checkpoints/last/pretrained_model")
        self.voice = self._prepare_voice()

    def _prepare_robot(self):
        print("Preparing robot")
        camera_external_config = OpenCVCameraConfig(
            index_or_path=7,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        )

        camera_bot_config = OpenCVCameraConfig(
            index_or_path=1,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        )

        robot_config = SO101FollowerConfig(
            port="/dev/ttyACM0",
            cameras={"robo": camera_bot_config, "out": camera_external_config},
            id="movie",
        )
        robot = SO101Follower(robot_config)
        robot.connect()
        print("Robot ready")
        return robot

    def _prepare_yolo(self):
        #return None
        print("Preparing YOLO")
        yolo = YOLOModel(Path("model.pt"))
        print("YOLO ready")
        return yolo

    def _prepare_voice(self):
        print("Preparing voice")
        voiceIA = VoiceIA()
        voiceIA.listen()
        print("Voice ready")
        return voiceIA

    #
    #   Robot States
    #

    def set_state(self, state: RobotStates):
        if not isinstance(state, RobotStates):
            raise ValueError(f"Invalid state: {state}")
        self.state = state
        self._verify_voice()

    def _verify_voice(self):
        if self.state == (RobotStates.STANDBY):
            self.voice.listen()
        elif self.state in (
            RobotStates.WAIT_VOICE_RELEASE,
            RobotStates.RETRIEVE_OBJECT,
            RobotStates.STORE_OBJECT,
            RobotStates.GOTO_STANDBY,
        ):
            self.voice.stop()

    def standby(self):
        # Aguarda presenca de pecas da pessoa

        # Aguarda comando de voz
        if self.voice.request_is_ready():
            self.request = self.voice.read_request()
            self.voice.stop()
            if(self.request.lower()=="pegar"):
                self.pick()
                return
            # Move to next state
            self.set_state(RobotStates.RETRIEVE_OBJECT)

    def pick(self):
        image = self.robot_cam.async_read(timeout_ms=200)
        detected_class = self.yolo.predict_best_class(image)
        if detected_class != "none":
            # Move to next state
            self.request = class_to_request[detected_class]
            time.sleep(2)
            self.set_state(RobotStates.STORE_OBJECT)

    def store_object(self):
        # Store object
        self.robotIA.run_task(self.request, duration=60)
        time.sleep(1)
        openGripper(self.robot)
        time.sleep(1)
        # Move to next state
        self.set_state(RobotStates.GOTO_STANDBY)

    def retrieve_object(self):
        # Vai para o ponto antes de pegar objetos
        self.robotIA.run_task(self.request, duration=60)

        # Move to next state
        self.set_state(RobotStates.WAIT_VOICE_RELEASE)

    def wait_voice_release(self):
        # Aguarda comando de voz para soltar
        if self.voice.listen_to_drop():
            openGripper(self.robot)
            time.sleep(2)

            # Move to next state
            self.set_state(RobotStates.GOTO_STANDBY)

    def goto_standby(self):
        """Utiliza ângulos fixos dos motores para ir para a posicao de standby"""
        # Move to standby position
        self.voice.listen()
        pos_standby(self.robot)

        # Move to next state
        self.set_state(RobotStates.STANDBY)


def main():
    bot = StateRunner()
    try:
        bot.run_states()
    except KeyboardInterrupt:
        print("Program stopped")


if __name__ == "__main__":
    main()
