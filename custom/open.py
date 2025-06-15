

def openGripper(robot):
    action=robot.get_observation()
    action["gripper.pos"]=40.252454417952315
    robot.send_action(action)
