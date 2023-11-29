import gymnasium as gym
from gymnasium import spaces
import os, sys
import numpy as np
import melee
import math
from pathlib import Path
from dotenv import load_dotenv

class MeleeEnv(gym.Env):
    
    def __init__(self):
        super(MeleeEnv, self).__init__()
        
        load_dotenv(Path("../.env"))

        # Connect to emulator and run melee
        self._setup()

        self._define_actions()

        self.action_space = gym.spaces.Discrete(26)

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

        print("initial state", self.gamestate)

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
        # TOFIX: select CPU on first episode
        while self.gamestate.menu_state != melee.Menu.IN_GAME:
            self.gamestate = self.console.step()
            
            if self.gamestate is None:
                continue

            if self.gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
                if self.gamestate.players[1].character != melee.Character.CPTFALCON:
                    melee.MenuHelper.choose_character(
                        gamestate=self.gamestate,
                        controller=self.controller1,
                        character=melee.Character.CPTFALCON,
                        cpu_level=0, # level 0 means player
                        costume=0,
                        start=False,
                        swag=False
                    )
                else:
                    self.controller1.press_button(melee.Button.BUTTON_A)
                
                if self.gamestate.players[2].character != melee.Character.DK or self.gamestate.players[2].cpu_level != 9:
                    melee.MenuHelper.choose_character(
                        gamestate=self.gamestate,
                        controller=self.controller2,
                        character=melee.Character.DK,
                        cpu_level=9,
                        costume=0,
                        start=True,
                        swag=False
                    )
                else:
                    self.controller2.release_button(melee.Button.BUTTON_A)
                    self.controller2.flush()
                    self.controller2.release_all()
                
                print(self.controller2.current.button)
            
            if self.gamestate.menu_state in [melee.Menu.PRESS_START, melee.Menu.MAIN_MENU]:
                melee.MenuHelper.choose_versus_mode(self.gamestate, self.controller2)

            elif self.gamestate.menu_state == melee.Menu.STAGE_SELECT:
                melee.MenuHelper.choose_stage(melee.Stage.BATTLEFIELD, self.gamestate, self.controller2)

            # # Select character for agent
            # melee.MenuHelper.menu_helper_simple(
            #     gamestate=self.gamestate,
            #     controller=self.controller1,
            #     character_selected=melee.Character.CPTFALCON,
            #     stage_selected=melee.Stage.BATTLEFIELD,
            #     connect_code="",
            #     cpu_level=0, # level 0 means player
            #     costume=0,
            #     autostart=False,
            #     swag=False
            # )

            # # Select character for opponent
            # melee.MenuHelper.menu_helper_simple(
            #     gamestate=self.gamestate,
            #     controller=self.controller2,
            #     character_selected=melee.Character.DK,
            #     stage_selected=melee.Stage.BATTLEFIELD,
            #     connect_code="",
            #     cpu_level=9,
            #     costume=0,
            #     autostart=False,
            #     swag=False
            # )
            
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
        print('Setting up')
        ISO_PATH = os.getenv('ISO_PATH')
        SLIPPI_PATH = os.getenv('SLIPPI_PATH')
        print(ISO_PATH, SLIPPI_PATH)
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
            type=melee.ControllerType.STANDARD
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
        controller.tilt_analog(melee.Button.BUTTON_C, self.action_map[action]['c_stick'][0], self.action_map[action]['c_stick'][1])
        if self.action_map[action]['button'] is not None:
            controller.release_all()
            controller.press_button(self.action_map[action]['button'])

    def _get_obs(self):
        agent = self.gamestate.players[1]
        adversary = self.gamestate.players[2]
        
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
        # 4 Smash Attacks               (A + [Up, Down, Left, Right] main stick)  
        # 5 Tilt/Aerial Attacks         (A or [Up, Down, Left, Right] C stick)
        # 5 Special Attacks             (B, B + [Up, Down, Left, Right] main stick)
        # 4 Movement                    ([Up, Down, Left, Right] main stick)
        # 1 Jump                        (Y)
        # 5 Shield/Air Dodge/Roll       (L or L + [Up, Down, Left, Right] main stick)
        # 1 Grab                        (Z)
        ANALOG_UP = (0.5, 1.0)
        ANALOG_DOWN = (0.5, 0)
        ANALOG_LEFT = (0.0, 0.5)
        ANALOG_RIGHT= (1.0, 0.5)
        ANALOG_NEUTRAL = (0.5, 0.5)
        self.action_map = [
            # Smash Attacks
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up Smash                     | 0
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down Smash                 | 1
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left Smash                 | 2
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right Smash               | 3

            # Tilt/Aerial Attacks 
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Neutral Tilt            | 4
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_UP}, # Up Tilt                                       | 5
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_DOWN}, # Down Tilt                                   | 6
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_LEFT}, # Left Tilt                                   | 7
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_RIGHT}, # Right Tilt                                 | 8

            # Special Attacks
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Neutral Special         | 9
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up Special                   | 10
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down Special               | 11
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left Special               | 12
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right Special             | 13

            # Movement
            {'button': None, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up                                            | 14
            {'button': None, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down                                        | 15
            {'button': None, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left                                        | 16
            {'button': None, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right                                      | 17

            # Jump
            {'button': melee.Button.BUTTON_Y, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Jump                    | 18

            # Shield/Air Dodge/Roll
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Shield Neutral          | 19
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Shield Up                    | 20
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Shield Down                | 21
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Shield Left                | 22
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Shield Right              | 23

            # Grab
            {'button': melee.Button.BUTTON_Z, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Grab                    | 24

            # No Action
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # No Action                                | 25
        ]

    def _calc_reward(self, obs):
        # TODO: make this actually good

        reward = [0]
        if self.prev_obs is not None:
            dpo = self.prev_obs['adversary_percent']
            d_o = obs['adversary_percent']
            dpa = self.prev_obs['percent']
            d_a = obs['percent']
            reward += (dpo-d_o) * math.e ** (-0.1*d_o) - (dpa-d_a) * math.e ** (-0.1*d_a) # R = (d'_o - d_o)e^(-0.1*d_o)-(d'_a - d_a)e^(-0.1*d_a)

            if self.prev_obs['stock'] > obs['stock']:
                reward -= 100
            if self.prev_obs['adversary_stock'] > obs['adversary_stock']:
                reward += 100

        self.prev_obs = obs
        return reward[0]

    def quit(self):
        #  Lt-Rt-A-Start to force exit
        while self.gamestate.menu_state != melee.Menu.IN_GAME:
            pass
        self.controller1.release_all()
        self.controller1.press_button(melee.Button.BUTTON_START)
        self.controller1.release_button(melee.Button.BUTTON_START)        
        self.controller1.press_button(melee.Button.BUTTON_L)
        self.controller1.press_button(melee.Button.BUTTON_R)
        self.controller1.press_button(melee.Button.BUTTON_A)
        self.controller1.press_button(melee.Button.BUTTON_START)
        self.controller1.release_all()
        # melee.MenuHelper.skip_postgame(self.controller1, self.gamestate)

