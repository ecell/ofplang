#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import shutil
# import hashlib
import uuid
import uvicorn
import asyncio
import datetime
import traceback
from contextlib import asynccontextmanager

import json
import numpy
import yaml  # type: ignore

from logging import getLogger
logger = getLogger(__name__)

import logging
logging.basicConfig()
logging.getLogger(__name__).setLevel(level=logging.INFO)
logging.getLogger('ofplang').setLevel(level=logging.INFO)

load_dotenv(verbose=True)
APPFILES_UPLOAD_PATH = Path('./files')
DEFINITIONS_FILE = Path("./definitions.yaml") # Path(os.environ.get("DEFINITIONS_FILE"))
APP_QUEUE_CHECKING_TIME = 5

def ndarray_representer(dumper: yaml.Dumper, array: numpy.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(numpy.ndarray, ndarray_representer)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(obj)

from ofplang.base import Runner, Definitions, Protocol
from ofplang.base.store import MinioArtifactStore
from ofplang.executors import TecanFluentController, Operator
from ofplang.executors.tecan import setup
artifact_store = MinioArtifactStore("10.5.1.234:9000", access_key="minio", secret_key="minio123", bucket_name="test", secure=False)
logger.info("Connecting to FluentControl...")
fluent = setup()
TECAN_FLUENT_OPERATOR = Operator(simulation=True, fluent=fluent)
logger.info("Connected...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup app...")
    _ = TECAN_FLUENT_OPERATOR.start()
    asyncio.create_task(waiting_consumer())
    yield
    logger.info("Shutdown app...")

APP_QUEUE_WAITING = asyncio.Queue()
app = FastAPI(lifespan=lifespan)

@app.get("/definitions")
async def get_definitions():
    return FileResponse(path=DEFINITIONS_FILE, filename=DEFINITIONS_FILE.name)

@app.get("/waiting", description="Retrieve the number of queued jobs.")
async def app_waiting():
    return APP_QUEUE_WAITING.qsize()

@app.post("/queue/")
async def queue_job(protocol: UploadFile, input: UploadFile | None | str = None, simulation: bool | None = None):
    # This is a workaround for clicking "Send empty value" on SwaggerUI.
    # See https://github.com/fastapi/fastapi/discussions/10280
    if isinstance(input, str):
        assert input == '', input
        input = None

    simulation = True if simulation is None else simulation

    job_id = str(uuid.uuid4())
    artifacts = APPFILES_UPLOAD_PATH / job_id
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / 'protocol.yaml').open("wb") as buf:
        shutil.copyfileobj(protocol.file, buf)
    if input is not None:
        with (artifacts / 'inputs.yaml').open("wb") as buf:
            shutil.copyfileobj(input.file, buf)

    job = {
        "id": job_id,
        "artifacts": str(artifacts),
        "protocol": str(artifacts / 'protocol.yaml'),
        "input": str(artifacts / 'inputs.yaml'),
        "simulation": simulation,
        "date": str(datetime.datetime.now()),
        }
    APP_QUEUE_WAITING.put_nowait(job)

    return job

async def waiting_consumer():
    while True:
        while APP_QUEUE_WAITING.empty():
            logger.info("Waiting...")
            await asyncio.sleep(APP_QUEUE_CHECKING_TIME)

        job = await APP_QUEUE_WAITING.get()

        try:
            await run(protocol=Path(job['protocol']), input=Path(job['input']), definitions=DEFINITIONS_FILE, artifacts=Path(job['artifacts']))
        except RuntimeError as e:
            logger.info(f"Job [{job['id']}] failed.")
            logger.info(f"{traceback.format_exc()}")
        else:
            logger.info(f"Job [{job['id']}] succeeded.")
        finally:
            APP_QUEUE_WAITING.task_done()

async def run(protocol, input, definitions, artifacts):
    with input.open() as f:
        inputs = yaml.load(f, Loader=yaml.Loader)

    runner = Runner(Protocol(protocol), Definitions(definitions), artifact_store=artifact_store)
    outputs = await runner.run(inputs=inputs, executor=TecanFluentController(TECAN_FLUENT_OPERATOR))
    logger.info(f"{str(outputs)}")

    with Path(artifacts / 'outputs.yaml').open('w') as f:
        yaml.dump(outputs, f, indent=2)

# def calculate_hash(file: Path) -> str:
#     with file.open('rb') as buf:
#         md5_hash = hashlib.md5()
#         while chunk := buf.read(8192):  # 8kb
#             md5_hash.update(chunk)
#     md5_hex = md5_hash.hexdigest()
#     return md5_hex


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7000, log_level="debug")
    # uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
