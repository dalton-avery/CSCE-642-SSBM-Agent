import gymnasium as gym
from gymnasium import spaces
import os, sys
import numpy as np
import melee
import math
from pathlib import Path
from dotenv import load_dotenv

class MeleeEnv(gym.Env):

    def __init__(self, opponent=7, port=51441):
        super(MeleeEnv, self).__init__()
        load_dotenv(Path("./.env"))
        self.opponent = opponent
        self.port = port
        
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

        self.gamestate = None
        self.framedata = melee.framedata.FrameData()
        

    def step(self, action):
        assert self.action_space.contains(action)
        self.gamestate = self.console.step()
        self._take_action(self.agent_controller, action)
        obs = self._get_obs()
        reward = self._calc_reward(obs)
        done = self.gamestate.menu_state != melee.Menu.IN_GAME # this may not work
        
        return self._get_flat_state(obs), reward, done, False, {}

    def reset(self):
        if not self.gamestate: self._setup(self.port)
        self.gamestate = self.console.step()
        self._clear_inputs()
        while self.gamestate.menu_state != melee.Menu.IN_GAME:
            self.gamestate = self.console.step()

            if self.gamestate is None:
                continue

            # Get to character select
            melee.menuhelper.MenuHelper.choose_versus_mode(gamestate=self.gamestate, controller=self.agent_controller)
            
            # Select characters
            if (self.gamestate.menu_state in [melee.enums.Menu.CHARACTER_SELECT]):
                self._clear_inputs()
                melee.menuhelper.MenuHelper.choose_character(character=melee.enums.Character.CPTFALCON, gamestate=self.gamestate, controller=self.adversary_controller, cpu_level=self.opponent, costume=1, swag=False, start=False)
                melee.menuhelper.MenuHelper.choose_character(character=melee.enums.Character.CPTFALCON, gamestate=self.gamestate, controller=self.agent_controller, cpu_level=0, costume=2, swag=False, start=False)
                if (self.gamestate.players[self.adversary_controller.port].cpu_level == self.opponent): # ready to start
                    melee.menuhelper.MenuHelper.skip_postgame(controller=self.agent_controller, gamestate=self.gamestate) # spam start
            
            # Select stage
            elif (self.gamestate.menu_state in [melee.enums.Menu.STAGE_SELECT]):
                self._clear_inputs()
                melee.menuhelper.MenuHelper.choose_stage(gamestate=self.gamestate, controller=self.agent_controller, stage=melee.enums.Stage.BATTLEFIELD)

        self.prev_obs = None
        return self._get_flat_state(self._get_obs()), None

    def can_receive_action(self):
        agent = self.gamestate.players[self.agent_controller.port]
        return self.framedata.attack_state(melee.Character.CPTFALCON, agent.action, agent.action_frame) == melee.enums.AttackState.NOT_ATTACKING

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

    def _setup(self, port):
        ISO_PATH = os.getenv('ISO_PATH')
        SLIPPI_PATH = os.getenv('SLIPPI_PATH')

        self.console = melee.console.Console(
            path=SLIPPI_PATH,
            fullscreen=False,
            online_delay=0,
            slippi_port=port,
            blocking_input=True
        )

        self.adversary_controller = melee.controller.Controller(
            self.console,
            port=1,
            type=melee.ControllerType.STANDARD
        )

        self.agent_controller = melee.controller.Controller(
            self.console,
            port=2 if self.opponent != 0 else 4,
            type=melee.ControllerType.STANDARD if self.opponent != 0 else melee.ControllerType.GCN_ADAPTER
        )

        print(f"Attemping to connect on port {port}")

        self.console.run(iso_path=ISO_PATH)
        if not self.console.connect():
            print("ERROR: Failed to connect to the console.")
            sys.exit(-1)
        print("Console connected")

        if not self.adversary_controller.connect() or not self.agent_controller.connect():
            print("ERROR: Failed to connect the controllers.")
            sys.exit(-1)
        print("Controllers connected")

    def _take_action(self, controller, action):
        controller.release_all()
        if self.action_map[action]['button'] is not None:
            controller.press_button(self.action_map[action]['button'])
        controller.tilt_analog(melee.Button.BUTTON_MAIN, self.action_map[action]['main_stick'][0], self.action_map[action]['main_stick'][1])

    def _clear_inputs(self):
        self.agent_controller.release_all()
        self.adversary_controller.release_all()

    def _define_actions(self):
        # 4 Smash/Aerial Attacks        (A + [Up, Down, Left, Right] main stick)  
        # 5 Tilt Attacks                (A or [Up, Down, Left, Right] slight main stick)
        # 5 Special Attacks             (B, B + [Up, Down, Left, Right] main stick)
        # 4 Movement                    ([Up, Down, Left, Right] main stick)
        # 5 Shield/Air Dodge/Roll       (L or L + [Up, Down, Left, Right] main stick)
        # 1 Grab                        (Z)
        #
        ANALOG_DOWN = (0.5, 0)
        ANALOG_TILT_DOWN = (0.5, 0.35)
        ANALOG_TILT_UP = (0.5, 0.65)
        ANALOG_UP = (0.5, 1.0)
        ANALOG_LEFT = (0.0, 0.5)
        ANALOG_TILT_LEFT = (0.35, 0.5)
        ANALOG_NEUTRAL = (0.5, 0.5)
        ANALOG_TILT_RIGHT = (0.65, 0.5)
        ANALOG_RIGHT= (1.0, 0.5)
        
        self.action_map = [
            # Smash Attacks
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_UP}, # Up Smash / Up Aair           | 0
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_DOWN}, # Down Smash / Down Air      | 1
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_LEFT}, # Left Smash / Forward Air   | 2
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_RIGHT}, # Right Smash / Back Air    | 3

            # Tilt/Aerial Attacks 
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_NEUTRAL}, # Jab / Neutral Air       | 4
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_TILT_UP}, # Up Tilt                 | 5
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_TILT_DOWN}, # Down Tilt             | 6
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_TILT_LEFT}, # Left Tilt             | 7
            {'button': melee.enums.Button.BUTTON_A, 'main_stick': ANALOG_TILT_RIGHT}, # Right Tilt           | 8

            # Special Attacks
            {'button': melee.enums.Button.BUTTON_B, 'main_stick': ANALOG_NEUTRAL}, # Neutral Special         | 9
            {'button': melee.enums.Button.BUTTON_B, 'main_stick': ANALOG_UP}, # Up Special                   | 10
            {'button': melee.enums.Button.BUTTON_B, 'main_stick': ANALOG_DOWN}, # Down Special               | 11
            {'button': melee.enums.Button.BUTTON_B, 'main_stick': ANALOG_LEFT}, # Left Special               | 12
            {'button': melee.enums.Button.BUTTON_B, 'main_stick': ANALOG_RIGHT}, # Right Special             | 13

            # Movement
            {'button': None, 'main_stick': ANALOG_UP}, # Jump                                                | 14
            {'button': None, 'main_stick': ANALOG_DOWN}, # Down                                              | 15
            {'button': None, 'main_stick': ANALOG_LEFT}, # Left                                              | 16
            {'button': None, 'main_stick': ANALOG_RIGHT}, # Right                                            | 17

            # Shield/Air Dodge/Roll
            {'button': melee.enums.Button.BUTTON_L, 'main_stick': ANALOG_NEUTRAL}, # Shield Neutral          | 18
            {'button': melee.enums.Button.BUTTON_L, 'main_stick': ANALOG_UP}, # Shield Up                    | 19
            {'button': melee.enums.Button.BUTTON_L, 'main_stick': ANALOG_DOWN}, # Shield Down                | 20
            {'button': melee.enums.Button.BUTTON_L, 'main_stick': ANALOG_LEFT}, # Shield Left                | 21
            {'button': melee.enums.Button.BUTTON_L, 'main_stick': ANALOG_RIGHT}, # Shield Right              | 22

            # Grab
            {'button': melee.Button.BUTTON_Z, 'main_stick': ANALOG_NEUTRAL}, # Grab                          | 23

            # No Action
            {'button': None, 'main_stick': ANALOG_NEUTRAL}, # No Action                                      | 24
        ]

    def _get_obs(self):
        agent = self.gamestate.players[self.agent_controller.port]
        adversary = self.gamestate.players[self.adversary_controller.port]
        
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

    def _calc_reward(self, obs):
        reward = 0
        if self.prev_obs is not None and self.gamestate.menu_state == melee.Menu.IN_GAME:
            # # Observation data
            # [agent_percent_delta] = obs['percent'] - self.prev_obs['percent'] # MINIMIZE
            # [adversary_percent_delta] = obs['adversary_percent'] - self.prev_obs['adversary_percent'] # MAXIMIZE
            # agent_stage_dist = np.clip(abs(obs['position'][0]) - abs(melee.stages.EDGE_POSITION[melee.Stage.BATTLEFIELD]), 0.0, 50.0) # MINIMIZE
            # adversary_stage_dist = np.clip(abs(obs['adversary_position'][0]) - abs(melee.stages.EDGE_POSITION[melee.Stage.BATTLEFIELD]), 0.0, 50.0) # MAXIMIZE
            # agent_y_delta = np.clip(obs['position'][1] - self.prev_obs['position'][1], -5.0, 15.0)
            # agent_frame_count = np.clip(self.framedata.frame_count(character=melee.Character.CPTFALCON, action=melee.enums.Action(obs['action'][0])), 0.0, 50.0) # PENALIZE
            # agent_curr_frame = obs['action'][1]

            # # Multipliers
            # hitstun_multiplier = 1.25 if obs['adversary_state_remainder'][2] > 1 else 1.0
            # cooldown_multiplier = 0.1

            # # Sub rewards
            # damage_reward = hitstun_multiplier * abs(adversary_percent_delta) - abs(agent_percent_delta) # their percent * multipliers - our percent
            # distance_reward = 0.2 * abs(adversary_stage_dist) - 0.2 * abs(agent_stage_dist) # abs(their distance from center) * offstage mult * 2.0 - abs(our distance from center) * offstage mult
            # offstage_reward = agent_y_delta if obs['state'][2] else 0.0 # delta y reward if offstage
            # endlag_reward = -cooldown_multiplier * agent_frame_count if agent_curr_frame == 1 else 0.0 # punish on first frame for using moves with large animation frames
            
            # # Main rewards
            # stock_reward = 0
            # if self.prev_obs['adversary_stock'] > obs['adversary_stock']:
            #     stock_reward += 500
            # if self.prev_obs['stock'] > obs['stock']:
            #     stock_reward -= 500
            
            # recovery_reward = 0 # + 10 for making back on stage (-5 for letting enemy back on stage?)
            # if not obs['state'][2] and self.prev_obs['state'][2]:
            #     recovery_reward += 25
            # if not obs['adversary_state'][2] and self.prev_obs['adversary_state'][2]:
            #     recovery_reward -= 10
            
            # reward = damage_reward + distance_reward + offstage_reward + stock_reward + recovery_reward + endlag_reward

            # SIMPLIFY
            # damage
            const_p = 1/300
            [agent_percent_delta] = obs['percent'] - self.prev_obs['percent'] 
            [adversary_percent_delta] = (obs['adversary_percent'] - self.prev_obs['adversary_percent'])
            damage_reward = const_p * (adversary_percent_delta - agent_percent_delta)

            # get back on stage
            recovery_reward = 0
            if not obs['state'][2] and self.prev_obs['state'][2]:
                recovery_reward = 0.02

            # stocks
            stock_reward = 0
            if self.prev_obs['adversary_stock'] > obs['adversary_stock']:
                stock_reward += 1
            if self.prev_obs['stock'] > obs['stock']:
                stock_reward -= 1

            reward = damage_reward + recovery_reward + stock_reward

        self.prev_obs = obs

        return reward

