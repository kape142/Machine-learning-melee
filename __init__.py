import melee
import numpy as np
import argparse
import signal
import sys
import time
from gym import spaces

CheckGameStatus = False

class MeleeBot:
    def __init__(self, render=False, iso_path=None):
        self.CheckGameStatus = False
        self.action_space = spaces.Discrete(4)  # [stand still, fsmash left, fsmash right, shield]
        high = np.array([
            # self
            #999,  # percent damage
            4,  # stocks
            5,  # current action
            #10,  # x-pos
            # opponent
            #999,  # percent damage
            4,  # stocks
            5,  # current action [stand still, fsmash left, fsmash right, shield, other (tumble+down)]
            #10,  # x-pos
        ])

        low = np.array([
            # self
            #0,  # percent damage
            0,  # stocks
            0,  # current action
            #-10,  # x-pos
            # opponent
            #0,  # percent damage
            0,  # stocks
            0,  # current action
            #-10,  # x-pos
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None
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
                                 default=True)
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
        self.opponent_type = melee.enums.ControllerType.UNPLUGGED
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
        signal.signal(signal.SIGINT, self.signal_handler)

        # Run dolphin and render the output
        self.dolphin.run(render=render, iso_path=iso_path)

        # Plug our controller inargs
        #   Due to how named pipes work, this has to come AFTER running dolphin
        #   NOTE: If you're loading a movie file, don't connect the controller,
        #   dolphin will hang waiting for input and never receive it
        self.controller.connect()

    def step(self, action):
        global CheckGameStatus
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        prevstate = self.state

        # "step" to the next frame
        self.gamestate.step()
        if self.gamestate.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(self.gamestate.processingtime * 1000) + "ms to process.")
        reward = 0
        # What menu are we in?
        if self.gamestate.menu_state in [melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH]:
            if self.CheckGameStatus == False:
                print("======= GAME STARTED ========")
                self.CheckGameStatus = True
            if self.args.framerecord:
                self.framedata.recordframe(self.gamestate)
            # XXX: This is where your AI does all of its stuff!
            # This line will get hit once per frame, so here is where you read
            #   in the gamestate and decide what buttons to push on the controller
            self.state = self.update_state(self.gamestate.ai_state, self.gamestate.opponent_state)
            reward += self.perform_action(action, self.gamestate.ai_state.tolist()[5])
        # If we're at the character select screen, choose our character
        elif self.gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
            melee.menuhelper.choosecharacter(character=melee.enums.Character.FALCO,
                                             gamestate=self.gamestate,
                                             port=self.args.port,
                                             opponent_port=self.args.opponent,
                                             controller=self.controller,
                                             swag=True,
                                             start=True)
        # If we're at the postgame scores screen, spam START
        elif self.gamestate.menu_state == melee.enums.Menu.POSTGAME_SCORES:
            melee.menuhelper.skippostgame(controller=self.controller)
        # If we're at the stage select screen, choose a stage
        elif self.gamestate.menu_state == melee.enums.Menu.STAGE_SELECT:
            melee.menuhelper.choosestage(stage=melee.enums.Stage.FINAL_DESTINATION,
                                         gamestate=self.gamestate,
                                         controller=self.controller)
        # Flush any button presses queued up
        self.controller.flush()
        if self.log:
            self.log.logframe(self.gamestate)
            self.log.writeframe()

        reward = self.get_reward(self.state, prevstate)
        done = False
        info = "I am currently %s, my damage is %.0f and i have %.0f lives left" % \
               (melee.enums.Action(self.gamestate.ai_state.tolist()[5]).name, self.state[1], self.state[2])
        return np.array(self.state), reward, done, info

    def perform_action(self, action, anim_state):
        en = melee.enums.Action
        if en(anim_state) != en.STANDING and 1 <= action <= 2:
            #print(en(anim_state).name)
            self.controller.empty_input()
            return -1
        if en(anim_state) != en.STANDING and action == 3:
            #print(en(anim_state).name)
            self.controller.empty_input()
            return -1
        if action == 1:
            self.controller.tilt_analog(melee.enums.Button.BUTTON_C, 0, 0.5)
            return 0
        if action == 2:
            self.controller.tilt_analog(melee.enums.Button.BUTTON_C, 1, 0.5)
            return 0
        if action == 3:
            self.controller.press_shoulder(melee.enums.Button.BUTTON_L, 1)
            return 0
        self.controller.empty_input()
        return 0

    def get_reward(self, state, prevstate):
        reward = 0
        #reward -= max(state[0] - prevstate[0], 0)
        reward -= (prevstate[0] - state[0]) * 400
        # reward += max(state[4] - prevstate[4], 0)
        # reward += (prevstate[5] - state[5]) * 400
        return reward

    def update_state(self, ai_state=melee.gamestate.PlayerState(), opponent_state=melee.gamestate.PlayerState()):
        state = np.zeros(4)
        ai_list = ai_state.tolist()
        opp_list = opponent_state.tolist()
        state[0] = ai_list[3]   #stock
        state[1] = self.action_to_number(ai_list[5], ai_list[4])   # current action
        #state[2] = ai_list[0]   # x position

        state[2] = opp_list[3]
        state[3] = self.action_to_number(opp_list[5], opp_list[4])
        #state[5] = opp_list[0]
        return state



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

    def action_to_enum_name(self, action):
        return next(name for name, value in vars(melee.enums.Action).items() if value == action)

    def action_to_number(self, action, facing):
        en = melee.enums.Action
        if action == en.STANDING.value or action == en.TURNING.value:
            return 0
        if en.FSMASH_HIGH.value <= action <= en.FSMASH_LOW.value:
            return 1 + facing
        if en.SHIELD_START.value <= action <= en.SHIELD_REFLECT.value:
            return 3
        #print(action)
        return 4

    def reset(self):
        self.state = np.array([
            # self
            #0,  # percent damage
            0,  # stocks
            0,  # current action
            #0,  # x-pos
            # opponent
            #0,  # percent damage
            0,  # stocks
            0,  # current action
            #0,  # x-pos
        ])  # må kanskje byttes ut med å hente fra dolphin?
        # må sikkert gjøre MYE mer
        return self.state

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