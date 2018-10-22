from meleebot import MeleeBot

bot = MeleeBot(iso_path="/home/espen/Documents/LibMelee/Super Smash Bros. Melee (v1.02).iso")  # change to your path to melee v1.02 NTSC ISO
bot.reset()
rewardtot = 0
while True:
    action = bot.action_space.sample()
    obv, reward, done, _ = bot.step(action)
    print(obv)
    rewardtot += reward
    print(rewardtot)
