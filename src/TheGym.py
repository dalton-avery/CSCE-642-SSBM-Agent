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

        self._define_actions()

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
        done = self.gamestate.menu_state != melee.Menu.IN_GAME # this may not work
        
        return obs, reward, done, False, {}

    def reset(self):
        while self.gamestate.menu_state != melee.Menu.IN_GAME:
            self.gamestate = self.console.step()
            if self.gamestate is None:
                continue

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
        controller.tilt_analog(melee.Button.BUTTON_MAIN, action['main_stick'][0], action['main_stick'][1])
        controller.tilt_analog(melee.Button.BUTTON_C, action['c_stick'][0], action['c_stick'][1])
        if action['button'] is not None:
            controller.press_button(action['button'])

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
            'action': np.array([agent.action, agent.action_frame]), # action_frame
            
            'adversary_position': np.array([adversary.position]),
            'adversary_shield_strength': np.array([adversary.shield_strength]),
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
            'adversary_action': np.array([adversary.action, adversary.action_frame]) # action_frame
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
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up Smash
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down Smash
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left Smash
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right Smash

            # Tilt/Aerial Attacks 
            {'button': melee.Button.BUTTON_A, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Neutral Tilt
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_UP}, # Up Tilt
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_DOWN}, # Down Tilt
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_LEFT}, # Left Tilt
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_RIGHT}, # Right Tilt

            # Special Attacks
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Neutral Special
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up Special
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down Special
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left Special
            {'button': melee.Button.BUTTON_B, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right Special

            # Movement
            {'button': None, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Up
            {'button': None, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Down
            {'button': None, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Left
            {'button': None, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Right

            # Jump
            {'button': melee.Button.BUTTON_Y, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Jump

            # Shield/Air Dodge/Roll
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Shield Neutral
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_UP, 'c_stick': ANALOG_NEUTRAL}, # Shield Up
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_DOWN, 'c_stick': ANALOG_NEUTRAL}, # Shield Down
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_LEFT, 'c_stick': ANALOG_NEUTRAL}, # Shield Left
            {'button': melee.Button.BUTTON_L, 'main_stick': ANALOG_RIGHT, 'c_stick': ANALOG_NEUTRAL}, # Shield Right

            # Grab
            {'button': melee.Button.BUTTON_Z, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # Grab

            # No Action
            {'button': None, 'main_stick': ANALOG_NEUTRAL, 'c_stick': ANALOG_NEUTRAL}, # No Action
        ]