from meleebot import MeleeBot

bot = MeleeBot(iso_path=None)  # change to your path to melee v1.02 NTSC ISO
bot.reset()
while True:
    action = bot.action_space.sample()
    obv, reward, done, _ = bot.step(action)
    print(obv)


