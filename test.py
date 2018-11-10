from meleebot import MeleeBot
import time

bot = None
try:
    bot = MeleeBot(iso_path="melee.iso", player_control=False)  # change to your path to melee v1.02 NTSC ISO
    bot.reset()
    rewardtot = [0,0]
    print("Action space: ", bot.action_space.n)
    print("Observation space: ", bot.observation_space.shape)
    i = 0;
    while bot.in_game == False:
        action = bot.action_space.sample()
        action2 = bot.action_space.sample()
        obv, reward, done, info = bot.step(action, action2)
    while True:
        i += 1
        action = bot.action_space.sample()
        action2 = bot.action_space.sample()
        obv, reward, done, info = bot.step(action, action2)

        rewardtot += reward
        # print(rewardtot)
        if i % 100 == 0:
            print(info)
            print(obv)
            time.sleep(1)
except Exception as e:
    print(e)
    bot.dolphin.terminate()
    time.sleep(0.5)
    bot.dolphin.terminate()
    raise e
