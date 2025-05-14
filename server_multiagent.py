import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from mlagents_envs.environment import UnityEnvironment, ActionTuple
import argparse
import os
import json
import time
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model once globally
yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt if GPU allows
yolo_class_dict = { 0: 'person',1: 'bicycle',2: 'car',3: 'motorcycle',4: 'airplane',5: 'bus',6: 'train',7: 'truck',8: 'boat',9: 'traffic light',10: 'fire hydrant',11: 'stop sign',12: 'parking meter',13: 'bench',14: 'bird',15: 'cat',16: 'dog',17: 'horse',18: 'sheep',19: 'cow',20: 'elephant',21: 'bear',22: 'zebra',23: 'giraffe',24: 'backpack',25: 'umbrella',26: 'handbag',27: 'tie',28: 'suitcase',29: 'frisbee',30: 'skis',31: 'snowboard',32: 'sports ball',33: 'kite',34: 'baseball bat',35: 'baseball glove',36: 'skateboard',37: 'surfboard',38: 'tennis racket',39: 'bottle',40: 'wine glass',41: 'cup',42: 'fork',43: 'knife',44: 'spoon',45: 'bowl',46: 'banana',47: 'apple',48: 'sandwich',49: 'orange',50: 'broccoli',51: 'carrot',52: 'hot dog',53: 'pizza',54: 'donut',55: 'cake',56: 'chair',57: 'couch',58: 'potted plant',59: 'bed',60: 'dining table',61: 'toilet',62: 'tv',63: 'laptop',64: 'mouse',65: 'remote',66: 'keyboard',67: 'cell phone',68: 'microwave',69: 'oven',70: 'toaster',71: 'sink',72: 'refrigerator',73: 'book',74: 'clock',75: 'vase',76: 'scissors',77: 'teddy bear',78: 'hair drier',79: 'toothbrush'}
gt_class_dict={0:'nothing',1:'person'}

def extarct_detection_from_yolo(agent_ids, decision_steps):
    """
    Runs YOLO on obs[0] image for each agent.
    Returns a dictionary: {agent_id: [ [class, x, y, w, h], ... ] }
    Also displays the image with bounding boxes using OpenCV.
    """
    detection_dict = {}
    for agent_id in agent_ids:
        obs_image = decision_steps[agent_id].obs[0]  # RGB (H, W, C)

        # Resize and convert color
        obs_image = cv2.resize(obs_image, (300, 300))
        obs_image = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)

        # Ensure uint8 image
        if obs_image.dtype != np.uint8:
            obs_image = (obs_image * 255).astype(np.uint8)

        # Run YOLO
        results = yolo_model(obs_image, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                #w, h = x2 - x1, y2 - y1
                #cx, cy = x1 + w // 2, y1 + h // 2
                detections.append([cls, x1, y1, x2, y2])

                # Draw bounding box
                #cv2.rectangle(obs_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                #label = f"{cls}"
                #cv2.putText(obs_image, label, (x1, y1 - 5),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Display with bounding boxes
        #cv2.imshow(f"Agent {agent_id} Detection", obs_image)
        #cv2.waitKey(1)

        detection_dict[agent_id] = detections

    return detection_dict

def extarct_gt(agent_ids, decision_steps):
    """
    Returns a dictionary: {agent_id: [ [class, x, y, w, h], ... ] }
    Also draws the GT boxes (scaled) on the resized image and displays it using cv2.
    """
    gt_dict = {}
    gt_width = 300
    gt_height = 300

    display_width = 300
    display_height = 300

    scale_x = display_width / gt_width
    scale_y = display_height / gt_height
    
    for agent_id in agent_ids:
        obs_image = decision_steps[agent_id].obs[0]  # (H, W, C)
        obs_image = cv2.resize(obs_image, (display_width, display_height))
        obs_image = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)
        if obs_image.dtype != np.uint8:
            obs_image = (obs_image * 255).astype(np.uint8)
        
        obs_gt = decision_steps[agent_id].obs[1]  # Flat list
        obs_list = list(obs_gt)
        gt_boxes = []

        if len(obs_list) % 6 != 0:
            continue

        for i in range(0, len(obs_list), 6):
            cls,instance_id, cx, cy, w, h = obs_list[i:i + 6]

            # Scale from original GT size to resized image size
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            x1 = int(cx_scaled)
            y1 = int(cy_scaled)
            x2 = int(cx_scaled + (w_scaled))
            y2 = int(cy_scaled + (h_scaled))

            gt_boxes.append([int(cls),int(instance_id), int(x1), int(y1), int(x2), int(y2)])

            # Convert center-width-height to corner points
            

            # Draw GT bounding box
            cv2.rectangle(obs_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            label = f"GT {int(cls)}"
            cv2.putText(obs_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow(f"Agent {agent_id} GT", obs_image)
        cv2.waitKey(1)

        gt_dict[agent_id] = gt_boxes

    return gt_dict




def calculate_global_tp_reward(gt_dict, det_dict, iou_threshold=0.5):
    """
    Computes a global true positive rate (TPR) across all agents as a reward.
    Each true positive is a unique instance ID correctly detected by any agent.
    TPR is defined as: (unique TP instance IDs) / (total instance IDs in the environment)
    """
    def iou(bb_det, bb_gt):
        if (bb_gt[2] != bb_gt[0] and bb_gt[3] != bb_gt[1]):
            xx1 = np.maximum(bb_det[0], bb_gt[0])
            yy1 = np.maximum(bb_det[1], bb_gt[1])
            xx2 = np.minimum(bb_det[2], bb_gt[2])
            yy2 = np.minimum(bb_det[3], bb_gt[3])
            w = np.maximum(0., xx2 - xx1)
            h = np.maximum(0., yy2 - yy1)
            wh = w * h
            o = wh / ((bb_det[2]-bb_det[0])*(bb_det[3]-bb_det[1]) +
                      (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        else:
            o = 0
        return o

    detected_instance_ids = set()
    total_instance_ids = [int(x) for x in range(8)]

    for agent_id in gt_dict.keys():
        gt_boxes = gt_dict.get(agent_id, [])
        det_boxes = det_dict.get(agent_id, [])

        #for gt in gt_boxes:
        #    total_instance_ids.add(int(gt[1]))  # instance_id

        for det in det_boxes:
            det_class, det_box = int(det[0]), det[1:]
            for gt in gt_boxes:
                gt_class, instance_id, gt_box = int(gt[0]), int(gt[1]), gt[2:]

                if instance_id in detected_instance_ids:
                    continue  # Already counted as TP

                if gt_class_dict[gt_class] == yolo_class_dict[det_class] and iou(det_box, gt_box) >= iou_threshold:
                    detected_instance_ids.add(instance_id)
                    break  # Move to next detection

    total_instances = len(total_instance_ids)
    if total_instances == 0:
        return 0.0  # avoid division by zero

    reward = len(detected_instance_ids) / total_instances  # TPR

    return reward




def preprocess_multi_agent_obs(decision_steps, expected_agents=None, normalize=True):
    if expected_agents is None:
        agent_ids = list(decision_steps)
    else:
        agent_ids = expected_agents

    processed_obs = []

    for agent_id in agent_ids:
        obs = decision_steps[agent_id].obs[0]  # assume first obs is the image
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        obs = obs.transpose(2, 0, 1)
        if normalize:
            obs = obs.astype(np.float32) / 255.0
        processed_obs.append(obs)

    return np.stack(processed_obs)

def rgb_to_grayscale(obs_rgb):
    return cv2.cvtColor(obs_rgb, cv2.COLOR_BGR2GRAY) # (H, W)

def decode_action(index, action_dict, num_cameras):
  

    # Mapping from string to integer
    action_map = {'DN': 0, 'INC': 1, 'DEC': 2}
    
    # Get the action string
    action_str = action_dict.get(index)
    if not action_str:
        raise ValueError(f"Invalid index {index}")
    
    # Split and extract the action values
    parts = action_str.split('_')
    values = parts[1::2]  # ['INC', 'DEC', 'DN']
    
    # Take only first `num_cameras` and map to int
    selected = values[:num_cameras]
    action_list = [[action_map[val]] for val in selected]

    return np.array(action_list, dtype=np.int32)



def preprocess_all_agents_obs(decision_steps, normalize=False):
    obs_stack = []
    for agent_id in decision_steps.agent_id:
        obs_rgb = decision_steps[agent_id].obs[0]  # Assume first obs is RGB
        obs_gray = rgb_to_grayscale(obs_rgb)  # (H, W)
        if normalize:
            obs_gray = obs_gray.astype(np.float32) / 255.0
        obs_stack.append(obs_gray)
    stacked_obs = np.stack(obs_stack, axis=0)  # Shape: (N_agents, H, W)
    return (stacked_obs*255).astype(np.uint8)

#global_action_dic={0: 'THETA1_INC_THETA2_INC', 1: 'THETA1_INC_THETA2_DEC', 2: 'THETA1_INC_THETA2_DN', 3: 'THETA1_DEC_THETA2_INC', 4: 'THETA1_DEC_THETA2_DEC', 5: 'THETA1_DEC_THETA2_DN', 6: 'THETA1_DN_THETA2_INC', 7: 'THETA1_DN_THETA2_DEC', 8: 'THETA1_DN_THETA2_DN'}
#global_action_dic={0: "THETA1_DN", 1: "THETA1_DEC", 2: "THETA1_INC"}
global_action_dic={0: 'THETA1_INC_THETA2_INC_THETA3_INC', 1: 'THETA1_INC_THETA2_INC_THETA3_DEC', 2: 'THETA1_INC_THETA2_INC_THETA3_DN', 3: 'THETA1_INC_THETA2_DEC_THETA3_INC', 4: 'THETA1_INC_THETA2_DEC_THETA3_DEC', 5: 'THETA1_INC_THETA2_DEC_THETA3_DN', 6: 'THETA1_INC_THETA2_DN_THETA3_INC', 7: 'THETA1_INC_THETA2_DN_THETA3_DEC', 8: 'THETA1_INC_THETA2_DN_THETA3_DN', 9: 'THETA1_DEC_THETA2_INC_THETA3_INC', 10: 'THETA1_DEC_THETA2_INC_THETA3_DEC', 11: 'THETA1_DEC_THETA2_INC_THETA3_DN', 12: 'THETA1_DEC_THETA2_DEC_THETA3_INC', 13: 'THETA1_DEC_THETA2_DEC_THETA3_DEC', 14: 'THETA1_DEC_THETA2_DEC_THETA3_DN', 15: 'THETA1_DEC_THETA2_DN_THETA3_INC', 16: 'THETA1_DEC_THETA2_DN_THETA3_DEC', 17: 'THETA1_DEC_THETA2_DN_THETA3_DN', 18: 'THETA1_DN_THETA2_INC_THETA3_INC', 19: 'THETA1_DN_THETA2_INC_THETA3_DEC', 20: 'THETA1_DN_THETA2_INC_THETA3_DN', 21: 'THETA1_DN_THETA2_DEC_THETA3_INC', 22: 'THETA1_DN_THETA2_DEC_THETA3_DEC', 23: 'THETA1_DN_THETA2_DEC_THETA3_DN', 24: 'THETA1_DN_THETA2_DN_THETA3_INC', 25: 'THETA1_DN_THETA2_DN_THETA3_DEC', 26: 'THETA1_DN_THETA2_DN_THETA3_DN'}
class DetailedLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(DetailedLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.detailed_log_path = os.path.join(log_dir, "detailed_training_log.json")
        self.log_data = {
            "episodes": [],
            "timesteps": [],
            "rewards": [],
            "episode_lengths": []
        }

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals.get("rewards", [0])[0]
        self.current_episode_length += 1

        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            self.log_data["episodes"].append(len(self.episode_rewards))
            self.log_data["timesteps"].append(self.num_timesteps)
            self.log_data["rewards"].append(float(self.current_episode_reward))
            self.log_data["episode_lengths"].append(self.current_episode_length)

            self.current_episode_reward = 0
            self.current_episode_length = 0

            if len(self.episode_rewards) % 10 == 0:
                with open(self.detailed_log_path, 'w') as f:
                    json.dump(self.log_data, f, indent=2)
                avg_reward = sum(self.episode_rewards[-10:]) / 10
                avg_length = sum(self.episode_lengths[-10:]) / 10
                print(f"Episode {len(self.episode_rewards)}: Avg Reward={avg_reward:.4f}, Avg Length={avg_length:.1f}")

        return True


class UnitySingleCameraEnv_SB3(gym.Env):
    def __init__(self, file_name=None, debug_mode=False):
        super(UnitySingleCameraEnv_SB3, self).__init__()
        self.env = UnityEnvironment(file_name=file_name, no_graphics=True)
        self.num_agents=0
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.behavior_spec = self.env.behavior_specs[self.behavior_name]
        self.debug_mode = debug_mode
        self.episode_reward = 0
        self.step_count = 0
        self.episode_count = 0

        self.env.step()
        behavior_name = list(self.env.behavior_specs.keys())[0]
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        self.num_agents = len(decision_steps)
     
        obs_shape = self.behavior_spec.observation_specs[0].shape
        assert len(obs_shape) == 3, f"Expected a single image observation, got shape: {obs_shape}"
        
        obs_shape_transposed = (self.num_agents , obs_shape[0], obs_shape[1])
        print("state of the DQN is",obs_shape_transposed)
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape_transposed, dtype=np.uint8)

        assert self.behavior_spec.action_spec.is_discrete(), "DQN only supports discrete actions."
        N_DISCRETE_ACTIONS = len(global_action_dic.keys())
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.agent_id = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print("reset is triggered")
        if self.debug_mode:
            print(f"Episode {self.episode_count} completed with total reward: {self.episode_reward} after {self.step_count} steps")

        self.episode_reward = 0
        self.step_count = 0
        self.episode_count += 1

        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        #print(decision_steps.obs)
        #print(len(decision_steps.obs))
        #self.agent_id = list(decision_steps.agent_id)
        #rint("self.agent_id",list(decision_steps.agent_id))
        obs_img = preprocess_all_agents_obs(decision_steps)
        return obs_img, {}

    def step(self, action):
        done=False 
        self.step_count += 1
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        agent_ids = list(decision_steps.agent_id)
        self.num_agents=len(agent_ids)
        num_agents = len(agent_ids)
        
        action_final=decode_action(action,global_action_dic,self.num_agents)
        action_tuple = ActionTuple(discrete=action_final)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        #if self.agent_id in decision_steps:
        #    obs_all = decision_steps[self.agent_id].obs
        #    done = False
        #else:
        #    obs_all = terminal_steps[self.agent_id].obs
        #    done = True
    
        #obs_img = obs_all[0]
        #obs_extra = obs_all[1]
        #if obs_img.dtype != np.uint8:
        #    obs_img = (obs_img * 255).astype(np.uint8)
        #obs_img = obs_img.transpose(2, 0, 1)
        obs_img = preprocess_all_agents_obs(decision_steps)

        GT_List_all_agets= extarct_gt(agent_ids,decision_steps) 
        #print("gt",GT_List_all_agets)
        Gdetection_List_all_agets= extarct_detection_from_yolo(agent_ids,decision_steps)
        #print('detection',Gdetection_List_all_agets)
        reward = calculate_global_tp_reward(GT_List_all_agets,Gdetection_List_all_agets)
        #print('reward is',reward)
        self.episode_reward += reward

        if self.debug_mode and (self.step_count % 10 == 0 or done):
            print(f"Step {self.step_count}, Reward: {reward}, Episode Reward: {self.episode_reward}")

        if self.step_count > 500:
            done = True

        return obs_img, reward, done, False, {"episode_reward": self.episode_reward}

    def calculate_reward_from_obs(self, obs_extra):
        if not isinstance(obs_extra, (np.ndarray, list)):
            return 0.0
        obs_list = list(obs_extra)
        if all(v == 0 for v in obs_list):
            return 0.0
        if len(obs_list) % 6 != 0:
            if self.debug_mode:
                print(f"[WARN] Invalid obs_extra length: {len(obs_list)} (not divisible by 5)")
            return 0.0

        num_bboxes = len(obs_list) // 6
        tp_count = 0
        total_gt = 0

        for i in range(num_bboxes):
            bbox = obs_list[i * 6:(i + 1) * 6]
            obj_type,instance_id, x, y, width, height = bbox
            if obj_type == 1:
                total_gt += 1
                tp_count += 1

        reward = float(tp_count / total_gt) if total_gt > 0 else 0.0
        return reward if reward > 0 else -0.01

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()


def create_callbacks(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix='dqn_model',
        verbose=1
    )
    logging_callback = DetailedLoggingCallback(log_dir=log_dir, verbose=1)
    return CallbackList([checkpoint_callback, logging_callback])


def make_env(debug_mode=False):
    env = UnitySingleCameraEnv_SB3(debug_mode=debug_mode)
    env = TimeLimit(env, max_episode_steps=500)
    env = Monitor(env, filename="./logs/monitor.csv", info_keywords=("episode_reward",))
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unity ML-Agents Training Script")
    parser.add_argument("--mode", type=str, choices=["train"], default="train", help="Training mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if args.mode == "train":
        print("Launching SB3 DQN training mode...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_dir = f"./training_runs/{timestamp}"
        tensorboard_log = f"{base_dir}/tensorboard"
        log_dir = f"{base_dir}/models"

        os.makedirs(tensorboard_log, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        with open(f"{base_dir}/training_args.json", 'w') as f:
            json.dump(vars(args), f, indent=2)

        train_env = DummyVecEnv([lambda: make_env(debug_mode=args.debug)])
        train_env = VecMonitor(train_env, filename=os.path.join(log_dir, "vec_monitor.csv"))

        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

        model = DQN(
            "CnnPolicy",
            train_env,
            verbose=1,
            train_freq=(4, "step"),
            gradient_steps=1,
            gamma=0.99,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            target_update_interval=200,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=32,
            learning_rate=1e-4,
            tensorboard_log=tensorboard_log,
            device="cuda",
            policy_kwargs=dict(net_arch=[150, 150], normalize_images=True),
        )

        model.set_logger(new_logger)
        callbacks = create_callbacks(log_dir)

        print(f"Starting training. Logs will be saved to {base_dir}")
        model.learn(
            total_timesteps=50000,
            callback=callbacks,
            progress_bar=True,
            log_interval=1
        )

        final_model_path = os.path.join(base_dir, "final_model")
        model.save(final_model_path)
        print(f"Training complete. Final model saved to {final_model_path}")
        train_env.close()
