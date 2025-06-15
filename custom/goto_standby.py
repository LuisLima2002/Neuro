from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower


def pos_standby(robot):
    robot.send_action({'shoulder_pan.pos': 46.08766898811962, 'shoulder_lift.pos': 8.195329087048833, 'elbow_flex.pos': -10.854503464203233, 'wrist_flex.pos': 84.39542483660131, 'wrist_roll.pos': 0.6444960041247754, 'gripper.pos': 40.252454417952315})
    

if __name__=="__main__":
    print("go to standby")
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM1",
        id="movie",
    )

    robot = SO101Follower(robot_config)
    robot.connect()
    import time
    time.sleep(1)
    try:
        while True:
            pos_standby(robot)
    except KeyboardInterrupt:
        pass
        # robot.disconnect()    

