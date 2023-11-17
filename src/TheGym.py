import gymnasium as gym
from gymnasium import spaces
import os, sys
import numpy as np
import melee

class MeleeEnv(gym.Env):
    
    def __init__(self):
        super(MeleeEnv, self).__init__()
        
        # Connect to emulator and run melee
        self._setup()

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
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up Smash               | 0
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down Smash           | 1
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left Smash           | 2
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right Smash         | 3

            # Tilt/Aerial Attacks 
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Neutral Tilt      | 4
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_UP}, # Up Tilt                                 | 5
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_DOWN}, # Down Tilt                             | 6
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_LEFT}, # Left Tilt                             | 7
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_RIGHT}, # Right Tilt                           | 8

            # Special Attacks
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Neutral Special   | 9
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up Special             | 10
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down Special         | 11
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left Special         | 12
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right Special       | 13

            # Movement
            {'button': None, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up                                      | 14
            {'button': None, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down                                  | 15
            {'button': None, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left                                  | 16
            {'button': None, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right                                | 17

            # Jump
            {'button': melee.Button.BUTTON_Y, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Jump              | 18

            # Shield/Air Dodge/Roll
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Shield Neutral    | 19
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Shield Up              | 20
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Shield Down          | 21
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Shield Left          | 22
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Shield Right        | 23

            # Grab
            {'button': melee.Button.BUTTON_Z, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Grab              | 24

            # No Action
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # No Action                          | 25
        ]
        self.action_space = gym.spaces.Discrete(26)


        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.Box(low=-10000, high=10000, shape=(2,), dtype=np.float32), # GET MIN AND MAX VALUES FROM LIB MELEE
            'shield_strength': gym.spaces.Box(low=0, high=60, shape=(1,), dtype=np.float32),
            'speed': gym.spaces.Box(low=-10000, high=10000, shape=(5,), dtype=np.float32),    # speed_air_x_self, speed_ground_x_self, speed_x_attack, speed_y_attack, speed_y_self
            'state':  gym.spaces.MultiBinary(4), # facing, on_ground, on_stage, invulnerable
            'state_remainder': gym.spaces.Discrete(3), # jumps_left, invulnerable_left, hitstun_frames_left
            'stock': gym.spaces.Discrete(1),
            'action': gym.spaces.Discrete(2) # action_frame
        })
        self.gamestate = self.console.step()

        self.reset()
        

    def step(self, action):
        assert self.action_space.contains(action)
        self.gamestate = self.console.step()
        self._take_action(self.controller1, action)
        obs = self._get_obs()
        reward = self.calc_reward(obs)
        # get gamestate values
        # calculate reward
        # if determine if done
        
        return obs, reward

    def reset(self):
        while self.gamestate.menu_state != melee.Menu.IN_GAME:
            self.gamestate = self.console.step()
            if self.gamestate is None:
                continue

            
            if self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                pass
            else:
                melee.MenuHelper.menu_helper_simple(
                    gamestate=self.gamestate,
                    controller=self.controller1,
                    character_selected=melee.Character.FOX,
                    stage_selected=melee.Stage.YOSHIS_STORY,
                    connect_code="",
                    cpu_level=0,
                    costume=0,
                    autostart=True,
                    swag=False
                )

                melee.MenuHelper.menu_helper_simple(
                    gamestate=self.gamestate,
                    controller=self.controller2,
                    character_selected=melee.Character.BOWSER,
                    stage_selected=melee.Stage.YOSHIS_STORY,
                    connect_code="",
                    cpu_level=6,
                    costume=0,
                    autostart=True,
                    swag=False
                )
        print(self._get_obs())

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
            print('Pressing button:', self.action_map[action]['button'])
            controller.press_button(self.action_map[action]['button'])

    def _get_obs(self):
        agent = self.gamestate.players[1]
        adversary = self.gamestate.players[2]
        
        obs = {
            'position': np.array([agent.position]),
            'shield_strength': np.array([agent.shield_strength]),
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
            'action': np.array([agent.action, agent.action_frame]) # action_frame
        }

        return obs
    
    def calc_reward(self, obs):
        # detect agent death        : -1
        # detect adversary death    :  1
        # function of percent       : R = (d'_o - d_o)e^(-0.1*d_o)-(d'_a - d_a)e^(-0.1*d_a)
        return 