import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(verbose=True)
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DISCORD_GUILD_ID = os.environ.get("DISCORD_GUILD_ID")
DEFINITIONS_FILE = Path(os.environ.get("DEFINITIONS_FILE"))

from logging import getLogger
logger = getLogger(__name__)

# # This example requires the 'message_content' intent.
# import datetime
import discord
from discord import app_commands

MY_GUILD = discord.Object(id=DISCORD_GUILD_ID)

class MyClient(discord.Client):

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.tree.copy_global_to(guild=MY_GUILD)
        await self.tree.sync(guild=MY_GUILD)

intents = discord.Intents.default()
discord_client = MyClient(intents=intents)

@discord_client.event
async def on_ready():
    logger.info(f'Logged in as {discord_client.user} (ID: {discord_client.user.id})')

def get_channel(name: str):
    for channel in discord_client.get_all_channels():
        if channel.name == name:
            return channel
    else:
        logger.error(f"No channel found [{name}]")
        return None

# @discord_client.tree.command()
# async def hello(interaction: discord.Interaction):
#     """Says hello!"""
#     # await interaction.response.send_message(f'Hi, {interaction.user.mention}')
#     embed = discord.Embed(title="Hi!!", description=f"How are you, {interaction.user.mention}?")
#     await interaction.response.send_message(embed=embed)

class BasicView(discord.ui.View):
    
    def __init__(self, *, future=None, **kwargs):
        super().__init__(**kwargs)

        self.__future = future

    @discord.ui.button(label="Ready", style=discord.ButtonStyle.green)
    async def click_ready(self, interaction: discord.Interaction, button: discord.Button) -> None:
        msg = f"✅ Ready {str(datetime.datetime.now())}, {interaction.user.mention} ."
        await interaction.response.send_message(msg)
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True
                item.style = discord.ButtonStyle.grey
        await interaction.message.edit(view=self)
        self.__future.set_result(True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def click_cancel(self, interaction: discord.Interaction, button: discord.Button) -> None:
        msg = f"❌ Canceled {str(datetime.datetime.now())}, {interaction.user.mention} ."
        await interaction.response.send_message(msg)
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True
                item.style = discord.ButtonStyle.grey
        await interaction.message.edit(view=self)
        self.__future.set_result(False)

# @discord_client.tree.command()
# async def button(interaction: discord.Interaction) -> None:
#     await interaction.response.send_message(view=BasicView(timeout=180.0))  # timeout is given in seconds.

# # discord_client.run(TOKEN)

# print('======')

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import shutil
# import hashlib
import pprint
import uuid
import uvicorn
import asyncio
import datetime
import traceback

import json
import numpy
import yaml  # type: ignore

def ndarray_representer(dumper: yaml.Dumper, array: numpy.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(numpy.ndarray, ndarray_representer)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(obj)

UPLOAD_PATH = Path('./files')

QUEUE_WAITING = asyncio.Queue()

app = FastAPI()

@app.on_event("startup")
async def startup_event():  # this fucntion will run before the main API starts
    asyncio.create_task(discord_client.start(DISCORD_TOKEN))
    await asyncio.sleep(4)  # optional sleep for established connection with discord
    logger.info(f"{discord_client.user} has connected to Discord!")
    asyncio.create_task(waiting_consumer())

    from ofplang.executors.fluent import tecan_fluent_operator
    asyncio.create_task(tecan_fluent_operator())

@app.get("/definitions")
async def get_definitions():
    return FileResponse(path=DEFINITIONS_FILE, filename=DEFINITIONS_FILE.name)

@app.get("/waiting", description="Retrieve the number of queued jobs.")
async def app_waiting():
    return QUEUE_WAITING.qsize()

@discord_client.tree.command(name="waiting", description="Retrieve the number of queued jobs.")
async def discord_waiting(interaction: discord.Interaction):
    await interaction.response.send_message(f"{QUEUE_WAITING.qsize()}", ephemeral=True)

@app.post("/queue/")
async def queue_job(protocol: UploadFile, input: UploadFile | None | str = None, simulation: bool | None = None):
    # This is a workaround for clicking "Send empty value" on SwaggerUI.
    # See https://github.com/fastapi/fastapi/discussions/10280
    if isinstance(input, str):
        assert input == '', input
        input = None

    simulation = True if simulation is None else simulation

    job_id = str(uuid.uuid4())
    artifacts = UPLOAD_PATH / job_id
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / 'protocol.yaml').open("wb") as buf:
        shutil.copyfileobj(protocol.file, buf)
    if input is not None:
        with (artifacts / 'input.yaml').open("wb") as buf:
            shutil.copyfileobj(input.file, buf)

    job = {
        "id": job_id,
        "artifacts": str(artifacts),
        "protocol": str(artifacts / 'protocol.yaml'),
        "input": str(artifacts / 'input.yaml'),
        "simulation": simulation,
        "date": str(datetime.datetime.now()),
        }
    QUEUE_WAITING.put_nowait(job)

    return job

async def waiting_consumer():
    while True:
        while QUEUE_WAITING.empty():
            await asyncio.sleep(3)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = await QUEUE_WAITING.get()

        msg = f"A new job [{job['id']}] was submitted on {job['date']}."
        embed = discord.Embed(title=msg, description=pprint.pformat(job, depth=80, indent=4))
        await get_channel('sandbox').send(embed=embed, view=BasicView(timeout=None, future=future))
        result = await future

        try:
            await run(protocol=Path(job['protocol']), input=Path(job['input']), definitions=DEFINITIONS_FILE, artifacts=Path(job['artifacts']))
        except RuntimeError as e:
            logger.info(f"Job [{job['id']}] failed.")
            logger.info(f"{traceback.format_exc()}")
            await get_channel('sandbox').send(f"❌ Job [{job['id']}] failed on {str(datetime.datetime.now())}.")
        else:
            logger.info(f"Job [{job['id']}] succeeded.")
            await get_channel('sandbox').send(f"✅ Job [{job['id']}] successfully finished on {str(datetime.datetime.now())}.")
        finally:
            QUEUE_WAITING.task_done()

async def run(protocol, input, definitions, artifacts):
    from ofplang.base import Runner, Definitions, Protocol
    from ofplang.executors import TecanFluentSimulator

    with input.open() as f:
        inputs = yaml.load(f, Loader=yaml.Loader)

    runner = Runner(Protocol(protocol), Definitions(definitions), executor=TecanFluentSimulator())
    experiment = await runner.run(inputs=inputs)
    outputs = experiment.output
    logger.info(f"{str(outputs)}")

    with Path(artifacts / 'output.yaml').open('w') as f:
        yaml.dump(outputs, f, indent=2)

# def calculate_hash(file: Path) -> str:
#     with file.open('rb') as buf:
#         md5_hash = hashlib.md5()
#         while chunk := buf.read(8192):  # 8kb
#             md5_hash.update(chunk)
#     md5_hex = md5_hash.hexdigest()
#     return md5_hex


if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(level=logging.INFO)
    logging.getLogger('ofplang').setLevel(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
