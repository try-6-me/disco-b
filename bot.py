import gc
import os
import random
from dotenv import load_dotenv

from random import seed
from random import randint

from discord.ext import commands
import discord
import torch
from generator_disco.generator import GeneratorDisco
from generator_ld.generator import GeneratorLatentDiffusion
from manager.chain.chain import Chain

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# 2
bot = commands.Bot(command_prefix='!')

chain = Chain()

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')


@bot.command(name='make', help='Makes art.')
async def make(ctx, prompt: str, input_seed:str=""):
    print ("making " + prompt)
    filename = chain.run_chain(prompt)
    await ctx.send(file=discord.File("static/output/" + filename))
    
    
bot.run(TOKEN)
