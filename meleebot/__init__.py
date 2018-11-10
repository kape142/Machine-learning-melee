import melee
import numpy as np
import argparse
import signal
import sys
import time
from gym import spaces


class MeleeBot:
    def __init__(self, render=False, iso_path=None, player_control=True):
        self.in_game = False
        self.done = False
        self.quitting = 0
        self.action_list = [
            lambda c: c.simple_press(0.5, 0.5, melee.enums.Button.BUTTON_MAIN), # 0: Analog stick center
            lambda c: c.simple_press(0.5, 0, melee.enums.Button.BUTTON_MAIN),   # 1: Analog stick down
            lambda c: c.simple_press(0, 0.5, melee.enums.Button.BUTTON_MAIN),   # 2: Analog stick left
            lambda c: c.simple_press(1, 0.5, melee.enums.Button.BUTTON_MAIN),   # 3: Analog stick right
            lambda c: c.simple_press(0.5, 0.5, melee.enums.Button.BUTTON_A),    # 4: Analog stick center + A button
            lambda c: c.simple_press(0.5, 0, melee.enums.Button.BUTTON_A),      # 5: Analog stick down + A button
            lambda c: c.simple_press(1, 0.5, melee.enums.Button.BUTTON_A),      # 6: Analog stick right + A button
            lambda c: c.simple_press(0, 0.5, melee.enums.Button.BUTTON_A),      # 7: Analog stick left  + A button
            lambda c: c.simple_press(0.5, 0.5, melee.enums.Button.BUTTON_R),    # 8: Analog stick center + R button
            lambda c: c.simple_press(0.5, 0, melee.enums.Button.BUTTON_R),      # 9: Analog stick down + R button
            lambda c: c.simple_press(1, 0.5, melee.enums.Button.BUTTON_R),      # 10: Analog stick right + R button
            lambda c: c.simple_press(0, 0.5, melee.enums.Button.BUTTON_R),      # 11: Analog stick left + R button
            lambda c: c.simple_press(0.5, 0.5, melee.enums.Button.BUTTON_Z),    # 12: Analog stick center + A button & R button
        ]

        self.lrastart_list = [
            lambda c: c.empty_input(),
            lambda c: {},
            lambda c: c.press_button(melee.enums.Button.BUTTON_START),
            lambda c: {},
            lambda c: {},
            lambda c: {},
            lambda c: c.press_shoulder(melee.enums.Button.BUTTON_L, 1),
            lambda c: {},
            lambda c: c.press_button(melee.enums.Button.BUTTON_L),
            lambda c: c.press_button(melee.enums.Button.BUTTON_A),
            lambda c: {},
            lambda c: {},
            lambda c: c.press_shoulder(melee.enums.Button.BUTTON_R, 1),
            lambda c: {},
            lambda c: c.press_button(melee.enums.Button.BUTTON_R),
            lambda c: c.empty_input()
        ]

        self.action_space = spaces.Discrete(len(self.action_list))  #
        high = np.array([
            27,  # current action [stand still, fsmash left, fsmash right, shield, other (tumble+down)]
            27,  # current opponent action
            20,  # discretized distance from opponent, 10 is immediate proximity, 0 is max left, 20 max right
            10,  # current x position of self
            10,  # current x position of opponent
        ])

        low = np.array([
            # self
            -1,   # current action
            -1,   # current opponent action
            0,    # distance
            -10,  # current x position of self
            -10,  # current x position of opponent
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.int)
        self.state = None
        self.state2 = None
        self.rewardstate = None
        self.rewardstate2 = None
        self.chain = None

        self.parser = argparse.ArgumentParser(description='Example of libmelee in action')
        self.parser.add_argument('--port', '-p', type=self.check_port,
                                 help='The controller port your AI will play on',
                                 default=2)
        self.parser.add_argument('--opponent', '-o', type=self.check_port,
                                 help='The controller port the opponent will play on',
                                 default=1)
        self.parser.add_argument('--live', '-l',
                                 help='The opponent is playing live with a GCN Adapter',
                                 default=player_control)
        self.parser.add_argument('--debug', '-d', action='store_true',
                                 help='Debug mode. Creates a CSV of all game state')
        self.parser.add_argument('--framerecord', '-r', default=False, action='store_true',
                                 help='(DEVELOPMENT ONLY) Records frame data from the match, stores into framedata.cs~/v.')

        self.args = self.parser.parse_args()

        self.log = None
        if self.args.debug:
            self.log = melee.logger.Logger()

        self.framedata = melee.framedata.FrameData(self.args.framerecord)

        # Options here are:
        #   "Standard" input is what dolphin calls the type of input that we use
        #       for named pipe (bot) input
        #   GCN_ADAPTER will use your WiiU adapter for live human-controlled play
        #   UNPLUGGED is pretty obvious what it means
        self.opponent_type = melee.enums.ControllerType.STANDARD
        if self.args.live:
            self.opponent_type = melee.enums.ControllerType.GCN_ADAPTER

        # Create our Dolphin object. This will be the primary object that we will interface with
        self.dolphin = melee.dolphin.Dolphin(ai_port=self.args.port,
                                             opponent_port=self.args.opponent,
                                             opponent_type=self.opponent_type,
                                             logger=self.log)
        # Create our GameState object for the dolphin instance
        self.gamestate = melee.gamestate.GameState(self.dolphin)
        # Create our Controller object that we can press buttons on
        self.controller = melee.controller.Controller(port=self.args.port, dolphin=self.dolphin)
        self.controller2 = melee.controller.Controller(port=self.args.opponent, dolphin=self.dolphin)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Run dolphin and render the output
        self.dolphin.run(render=render, iso_path=iso_path, batch=True)

        # Plug our controller inargs
        #   Due to how named pipes work, this has to come AFTER running dolphin
        #   NOTE: If you're loading a movie file, don't connect the controller,
        #   dolphin will hang waiting for input and never receive it
        self.controller.connect()
        if not player_control:
            self.controller2.connect()

    def step(self, action, action2=None):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert action2 is None or self.action_space.contains(action2), "%r (%s) invalid" % (action, type(action))
        assert action2 is not None or self.args.live
        prevrewardstate = self.rewardstate
        prevrewardstate2 = self.rewardstate2

        # "step" to the next frame
        self.gamestate.step()
        if self.gamestate.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(self.gamestate.processingtime * 1000) + "ms to process.")
        reward = 0
        reward2 = 0
        done = False
        ai_list = self.gamestate.ai_state.tolist()
        opp_list = self.gamestate.opponent_state.tolist()
        # What menu are we in?
        self.in_game = False
        if self.gamestate.menu_state in [melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH]:
            self.in_game = True
            if self.args.framerecord:
                self.framedata.recordframe(self.gamestate)
            self.state = self.update_state(ai_list, opp_list)
            self.rewardstate = self.update_rewardstate(ai_list, opp_list)
            if not self.args.live:
                self.state2 = self.update_state(opp_list, ai_list)
                self.rewardstate2 = self.update_rewardstate(opp_list, ai_list)
            if self.done:
                if self.quitting < len(self.lrastart_list):
                    self.lrastart_list[self.quitting](self.controller)
                    self.quitting += 1
            else:
                self.action_list[action](self.controller)
                if action2 is not None:
                    self.action_list[action2](self.controller2)
        # If we're at the character select screen, choose our character
        elif self.gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
            melee.menuhelper.choosecharacter(character=melee.enums.Character.FALCO,
                                             gamestate=self.gamestate,
                                             port=self.args.port,
                                             opponent_port=self.args.opponent,
                                             controller=self.controller,
                                             swag=False,
                                             start=True)
            if not self.args.live:
                melee.menuhelper.choosecharacter(character=melee.enums.Character.FALCO,
                                                 gamestate=self.gamestate,
                                                 port=self.args.opponent,
                                                 opponent_port=self.args.port,
                                                 controller=self.controller2,
                                                 swag=False,
                                                 start=True)
        # If we're at the postgame scores screen, spam START
        elif self.gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
            melee.menuhelper.skippostgame(controller=self.controller)
            if self.done:
                done = True
                self.done = False
                self.quitting=0

            if not self.args.live:
                melee.menuhelper.skippostgame(controller=self.controller2)
        # If we're at the stage select screen, choose a stageopponent
        elif self.gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
            melee.menuhelper.choosestage(stage=melee.enums.Stage.FINAL_DESTINATION,
                                         gamestate=self.gamestate,
                                         controller=self.controller)
        # Flush any button presses queued up
        self.controller.flush()
        if not self.args.live:
            self.controller2.flush()
        if self.log:
            self.log.logframe(self.gamestate)
            self.log.writeframe()

        reward += self.get_reward(self.rewardstate, prevrewardstate)
        if not self.args.live:
            reward2 += self.get_reward(self.rewardstate2, prevrewardstate2)
        info = "I am currently doing %s, which corresponds to action #%0.f, " \
               "my opponent is doing %s, which corresponds to action #%0.f, " \
               "and the relative distance to my opponent is %.0f" % \
               (melee.enums.Action(ai_list[5]).name, self.state[0],
                melee.enums.Action(opp_list[5]).name, self.state[1],
                self.state[2])
        info = ai_list[5], opp_list[5]
        return [self.state, self.state2], [reward, reward2], done, info

    def get_reward(self, state, prevstate):
        reward = -0.0166667  # ~ -1.0 each second
        reward -= max(state[0] - prevstate[0], 0)  # percent self
        reward -= (prevstate[1] - state[1]) * 100  # stock self
        reward += max(state[3] - prevstate[3], 0) * 2  # percent opponent
        reward += (prevstate[4] - state[4]) * 100  # stock opponent

        if  abs(state[2]) > 7:
            reward -= 0.005        # -0,3 each second self is at the edge
        if abs(state[5]) > 7:
            reward += 0.005        # 0,3 each second opponent is at the edge
        return reward

    def update_state(self, ai_list, opp_list):
        state = np.zeros(5)
        state[0] = self.action_to_number(ai_list[5])
        state[1] = self.action_to_number(opp_list[5])
        state[2] = self.discretize_distance(ai_list[0], opp_list[0])
        state[3] = self.discretize_position(ai_list[0])
        state[4] = self.discretize_position(opp_list[0])
        return state

    def update_rewardstate(self, ai_list, opp_list):
        state = np.zeros(6)
        state[0] = ai_list[2]                            # percent
        state[1] = ai_list[3]                            # stock
        state[2] = self.discretize_position(ai_list[0])  # Discretized x pos
        state[3] = opp_list[2]
        state[4] = opp_list[3]
        state[5] = self.discretize_position(opp_list[0])
        return state

    # Skal vi gjøre det samme her KP som du gjorde med discretize_distance?
    def discretize_position(self, position):  # Mapper verdi til [-10, 10]
        sigdist = np.sign(position)
        if abs(position) > 100:
            discretized_position = 10 * sigdist
        else:
            discretized_position = int(position / 10)
        return discretized_position

    def discretize_distance(self, xpos_self, xpos_opp):
        distance = xpos_self - xpos_opp
        absdist = abs(distance)
        sigdist = np.sign(distance)
        if absdist > 100:
            return 10 + 10 * sigdist
        sqdist = np.floor(np.sqrt(absdist))
        return 10 + sqdist * sigdist

    def state_to_action_name(self, action):
        if action == 0:
            return "standing still"
        if action == 1:
            return "kicking left"
        if action == 2:
            return "kicking right"
        if action == 3:
            return "shielding"
        return "doing something weird"

    def action_to_number(self, action_value):
        en = melee.enums.Action
        action = en(action_value)
        if action in [en.DEAD_DOWN, en.ON_HALO_DESCENT, en.ENTRY, en.ENTRY_START, en.ENTRY_END]:
            return 0
        if action in [en.STANDING, en.WALK_SLOW, en.TURNING]:
            return 1
        if en.DAMAGE_HIGH_1.value <= action.value <= en.DAMAGE_FLY_ROLL.value:
            return 2
        if en.SHIELD_START.value <= action.value <= en.SHIELD_REFLECT.value:
            return 3
        if action == en.DASHING:
            return 4
        if action in [en.CROUCH_START, en.CROUCHING, en.CROUCH_END]:
            return 5
        if action in [en.LANDING, en.LANDING_SPECIAL]:
            return 6
        if action in [en.NEUTRAL_ATTACK_1, en.NEUTRAL_ATTACK_2, en.NEUTRAL_ATTACK_3]:
            return 7
        if action in [en.LOOPING_ATTACK_START, en.LOOPING_ATTACK_MIDDLE, en.LOOPING_ATTACK_END]:
            return 8
        if action == en.DASH_ATTACK:
            return 9
        if action in [en.FALLING, en.DEAD_FALL, en.TUMBLING]:
            return 10
        if action == en.FTILT_MID:
            return 11
        if action == en.FSMASH_MID:
            return 12
        if action == en.DOWNSMASH:
            return 13
        if action in [en.NAIR, en.FAIR, en.DAIR]:  # bare ok så lenge de egentlig ikke skal hoppe
            return 14
        if action in [en.NAIR_LANDING, en.FAIR_LANDING]:
            return 15
        if action in [en.TECH_MISS_UP, en.NEUTRAL_GETUP]:
            return 16
        if action in [en.GROUND_ATTACK_UP, en.DAMAGE_GROUND]:
            return 17
        if en.GRAB.value <= action.value <= en.GRAB_BREAK.value:
            return 18
        if en.GRAB_PULL.value <= action.value <= en.GRABBED.value:
            return 19
        if action == en.THROWN_FORWARD:
            return 20
        if action == en.THROWN_BACK:
            return 21
        if action == en.THROWN_DOWN:
            return 22
        if action == en.ROLL_FORWARD:
            return 23
        if action == en.ROLL_BACKWARD:
            return 24
        if action == en.SPOTDODGE:
            return 25
        if action == en.AIRDODGE:
            return 26
        if en.THROW_FORWARD.value <= action.value <= en.THROW_DOWN.value:
            return 27
        return -1

    def reset(self):
        self.state = np.zeros(5)
        self.state2 = np.zeros(5)
        self.rewardstate = np.zeros(6)
        self.rewardstate2 = np.zeros(6)
        return [self.state, self.state2]

    def check_port(self, value):
        ivalue = int(value)
        if ivalue < 1 or ivalue > 4:
            raise argparse.ArgumentTypeError("%s is an invalid controller port. \
             Must be 1, 2, 3, or 4." % value)
        return ivalue

    def signal_handler(self, signal, frame):
        self.dolphin.terminate()
        if self.args.debug:
            self.log.writelog()
            print("")  # because the ^C will be on the terminal
            print("Log file created: " + self.log.filename)
        print("Shutting down cleanly...")
        if self.args.framerecord:
            self.framedata.saverecording()
        sys.exit(0)
