from meleebot import MeleeBot
import time

bot = None
try:
    bot = MeleeBot(iso_path="melee.iso", player_control=False)  # change to your path to melee v1.02 NTSC ISO
    bot.reset()
    rewardtot = [0,0]
    print("Action space: ", bot.action_space.n)
    print("Observation space: ", bot.observation_space.shape)
    while True:
        action = bot.action_space.sample()
        action2 = bot.action_space.sample()
        obv, reward, done, info = bot.step(action, action2)
        print(info)
        print(obv)
        rewardtot += reward
        # print(rewardtot)
except Exception as e:
    print(e)
    bot.dolphin.terminate()
    time.sleep(0.5)
    bot.dolphin.terminate()
    raise e
