import gymnasium as gym
from gymnasium import spaces
import os, sys
import numpy as np
import melee
import math
from pathlib import Path
from dotenv import load_dotenv

class MeleeEnv(gym.Env):
    
    def __init__(self, opponent=9):
        super(MeleeEnv, self).__init__()
        load_dotenv(Path("./.env"))
        self.opponent = opponent
        
        # Connect to emulator and run melee
        self._setup()

        self._define_actions()

        self.action_space = gym.spaces.Discrete(25)

        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.Box(low=-10000, high=10000, shape=(2,), dtype=np.float32), # GET MIN AND MAX VALUES FROM LIB MELEE
            'shield_strength': gym.spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
            'percent': gym.spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'speed': gym.spaces.Box(low=-10000, high=10000, shape=(5,), dtype=np.float32),    # speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
            'state':  gym.spaces.MultiBinary(4), # facing, on_ground, on_stage, invulnerable
            'state_remainder': gym.spaces.Discrete(3), # jumps_left, invulnerable_left, hitstun_frames_left
            'stock': gym.spaces.Discrete(1),
            'action': gym.spaces.Discrete(2), # action_frame

            'adversary_position': gym.spaces.Box(low=-10000, high=10000, shape=(2,), dtype=np.float32), # GET MIN AND MAX VALUES FROM LIB MELEE
            'adversary_shield_strength': gym.spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
            'adversary_percent': gym.spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'adversary_speed': gym.spaces.Box(low=-10000, high=10000, shape=(5,), dtype=np.float32),    # speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
            'adversary_state':  gym.spaces.MultiBinary(4), # facing, on_ground, on_stage, invulnerable
            'adversary_state_remainder': gym.spaces.Discrete(3), # jumps_left, invulnerable_left, hitstun_frames_left
            'adversary_stock': gym.spaces.Discrete(1),
            'adversary_action': gym.spaces.Discrete(2) # action_frame
        })

        self.gamestate = self.console.step()

        self.reset()
        

    def step(self, action):
        assert self.action_space.contains(action)
        self.gamestate = self.console.step()
        self._take_action(self.controller1, action)
        obs = self._get_obs()
        reward = self._calc_reward(obs)
        done = self.gamestate.menu_state != melee.Menu.IN_GAME # this may not work
        
        return self._get_flat_state(obs), reward, done, False, {}

    def reset(self):
        while self.gamestate.menu_state != melee.Menu.IN_GAME:
            self.gamestate = self.console.step()
            if self.gamestate is None:
                continue

            melee.MenuHelper.menu_helper_simple(
                gamestate=self.gamestate,
                controller=self.controller1,
                character_selected=melee.Character.CPTFALCON,
                stage_selected=melee.Stage.BATTLEFIELD,
                connect_code="",
                cpu_level=0,
                costume=0,
                autostart=self.opponent, # false if hmn
                swag=False
            )

            melee.MenuHelper.menu_helper_simple(
                gamestate=self.gamestate,
                controller=self.controller2,
                character_selected=melee.Character.DK,
                stage_selected=melee.Stage.BATTLEFIELD,
                connect_code="",
                cpu_level=self.opponent,
                costume=0,
                autostart=self.opponent, # false if hmn
                swag=False
            )
        self.prev_obs = None
        return self._get_flat_state(self._get_obs()), None

    def get_obs_shape(self):
        input_size = 0

        for key, space in self.observation_space.spaces.items():
            if isinstance(space, gym.spaces.Discrete):
                input_size += space.n
            else:
                input_size += np.prod(space.shape)

        return input_size
    
    def _get_flat_state(self, obs):
        flat_state = []
        for key, space in obs.items():
            for val in space:
                flat_state.append(val)
        return np.array(flat_state, dtype=np.float32)

    def _setup(self):
        ISO_PATH = os.getenv('ISO_PATH')
        SLIPPI_PATH = os.getenv('SLIPPI_PATH')

        self.console = melee.console.Console(
            path=SLIPPI_PATH,
            fullscreen=False
        )

        self.controller1 = melee.controller.Controller(
            self.console,
            port=1,
            type=melee.ControllerType.STANDARD
        )

        self.controller2 = melee.controller.Controller(
            self.console,
            port=2,
            type=melee.ControllerType.GCN_ADAPTER if self.opponent == 0 else melee.ControllerType.STANDARD
        )

        self.console.run(iso_path=ISO_PATH)
        print("Connecting to console...")
        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            sys.exit(-1)
        print("Console connected")

        print("Connecting controllers to console...")
        if not self.controller1.connect() or not self.controller2.connect():
            print("ERROR: Failed to connect the controllers.")
            sys.exit(-1)
        print("Controllers connected")

    def _take_action(self, controller, action):
        controller.tilt_analog(melee.Button.BUTTON_MAIN, self.action_map[action]['main_stick'][0], self.action_map[action]['main_stick'][1])
        if self.action_map[action]['button'] is not None:
            controller.release_all()
            controller.press_button(self.action_map[action]['button'])

    def _get_obs(self):
        agent = self.gamestate.players[1]
        adversary = self.gamestate.players[2] if self.opponent != 0 else self.gamestate.players[4]
        
        obs = {
            'position': np.array([agent.position.x, agent.position.y]),
            'shield_strength': np.array([agent.shield_strength]),
            'percent': np.array([agent.percent]),
            'speed': np.array(
                [agent.speed_air_x_self,
                agent.speed_ground_x_self,
                agent.speed_y_self,
                agent.speed_x_attack,
                agent.speed_y_attack]
            ),    # speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
            'state':  np.array(
                [agent.facing,
                agent.on_ground,
                agent.off_stage,
                agent.invulnerable]
            ), # facing, on_ground, on_stage, invulnerable
            'state_remainder': np.array(
                [agent.jumps_left,
                agent.invulnerability_left,
                agent.hitstun_frames_left]
            ), # jumps_left, invulnerability_left, hitstun_frames_left
            'stock': np.array([agent.stock]),
            'action': np.array([agent.action.value, agent.action_frame]), # action_frame
            
            'adversary_position': np.array([adversary.position.x, adversary.position.y]),
            'adversary_shield_strength': np.array([adversary.shield_strength]),
            'adversary_percent': np.array([adversary.percent]),
            'adversary_speed': np.array(
                [adversary.speed_air_x_self,
                adversary.speed_ground_x_self,
                adversary.speed_y_self,
                adversary.speed_x_attack,
                adversary.speed_y_attack]
            ),    # speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
            'adversary_state':  np.array(
                [adversary.facing,
                adversary.on_ground,
                adversary.off_stage,
                adversary.invulnerable]
            ), # facing, on_ground, on_stage, invulnerable
            'adversary_state_remainder': np.array(
                [adversary.jumps_left,
                adversary.invulnerability_left,
                adversary.hitstun_frames_left]
            ), # jumps_left, invulnerability_left, hitstun_frames_left
            'adversary_stock': np.array([adversary.stock]),
            'adversary_action': np.array([adversary.action.value, adversary.action_frame]) # action_frame
        }

        return obs
    
    def _define_actions(self):
        # 4 Smash/Aerial Attacks        (A + [Up, Down, Left, Right] main stick)  
        # 5 Tilt Attacks                (A or [Up, Down, Left, Right] C stick)
        # 5 Special Attacks             (B, B + [Up, Down, Left, Right] main stick)
        # 4 Movement                    ([Up, Down, Left, Right] main stick)
        # 5 Shield/Air Dodge/Roll       (L or L + [Up, Down, Left, Right] main stick)
        # 1 Grab                        (Z)
        ANALOG_DOWN = (0.5, 0)
        ANALOG_TILT_DOWN = (0.5, 0.25)
        ANALOG_TILT_UP = (0.5, 0.75)
        ANALOG_UP = (0.5, 1.0)
        ANALOG_LEFT = (0.0, 0.5)
        ANALOG_TILT_LEFT = (0.25, 0.5)
        ANALOG_NEUTRAL = (0.5, 0.5)
        ANALOG_TILT_RIGHT = (0.75, 0.5)
        ANALOG_RIGHT= (1.0, 0.5)
        
        self.action_map = [
            # Smash Attacks
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_UP}, # Up Smash / Up Aair           | 0
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_DOWN}, # Down Smash / Down Air      | 1
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_LEFT}, # Left Smash / Forward Air   | 2
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_RIGHT}, # Right Smash / Back Air    | 3

            # Tilt/Aerial Attacks 
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_NEUTRAL}, # Jab / Neutral Air       | 4
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_TILT_UP}, # Up Tilt                 | 5
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_TILT_DOWN}, # Down Tilt             | 6
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_TILT_LEFT}, # Left Tilt             | 7
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_TILT_RIGHT}, # Right Tilt           | 8

            # Special Attacks
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_NEUTRAL}, # Neutral Special         | 9
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_UP}, # Up Special                   | 10
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_DOWN}, # Down Special               | 11
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_LEFT}, # Left Special               | 12
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_RIGHT}, # Right Special             | 13

            # Movement
            {'button': None, 'main_stick': ANALOG_UP}, # Jump                                          | 14
            {'button': None, 'main_stick': ANALOG_DOWN}, # Down                                        | 15
            {'button': None, 'main_stick': ANALOG_LEFT}, # Left                                        | 16
            {'button': None, 'main_stick': ANALOG_RIGHT}, # Right                                      | 17

            # Shield/Air Dodge/Roll
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_NEUTRAL}, # Shield Neutral          | 18
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_UP}, # Shield Up                    | 19
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_DOWN}, # Shield Down                | 20
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_LEFT}, # Shield Left                | 21
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_RIGHT}, # Shield Right              | 22

            # Grab
            {'button': melee.Button.BUTTON_Z, 'main_stick': ANALOG_NEUTRAL}, # Grab                    | 23

            # No Action
            {'button': None, 'main_stick': ANALOG_NEUTRAL}, # No Action                                | 24
        ]

    def _calc_reward(self, obs):
        # TODO: make this actually good

        reward = 0
        if self.prev_obs is not None and self.gamestate.menu_state == melee.Menu.IN_GAME:
            [agent_delta] = obs['percent'] - self.prev_obs['percent']
            [adver_delta] = obs['adversary_percent'] - self.prev_obs['adversary_percent']
            prev_off_stage = self.prev_obs['state'][2]
            curr_off_stage = obs['state'][2]

            if self.prev_obs['adversary_stock'] == obs['adversary_stock'] and self.prev_obs['stock'] == obs['stock']:
                reward += abs(adver_delta) - abs(agent_delta)

            if self.prev_obs['adversary_stock'] > obs['adversary_stock']:
                reward += 1000

            if self.prev_obs['stock'] > obs['stock']:
                reward -= 1000
            
            if curr_off_stage:
                reward -= 1


        self.prev_obs = obs
        if round(reward) != 0:
            print(round(reward))

        return reward
    
    def skip_episode(self):
        while self.gamestate.menu_state == melee.Menu.IN_GAME:
            self.step(16)
