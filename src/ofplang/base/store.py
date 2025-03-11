#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

import dataclasses
import uuid
from pathlib import Path, PurePath
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import copy
from datetime import datetime
from typing import Any, ValuesView
import requests

from .protocol import EntityDescription, PortAddress

logger = getLogger(__name__)


@dataclasses.dataclass
class Location:
    id: str
    uri: str

class Handler:

    def __init__(self):
        pass

    def create_run(self, run_id: str, storage_address: str, metadata: dict | None) -> None:
        pass

    def finish_run(self, run_id: str, status: str, metadata: dict | None) -> None:
        pass

    def create_process(self, process_id: str, storage_address: str, metadata: dict | None) -> None:
        pass

    def finish_process(self, process_id: str, status: str, metadata: dict | None) -> None:
        pass

    def create_operation(self, operation_id: str, storage_address: str, metadata: dict | None) -> None:
        pass

    def finish_operation(self, operation_id: str, status: str, metadata: dict | None) -> None:
        pass

    def log_operation_text(self, operation_id: str, text: str, artifact_file: str) -> None:
        pass

class Store_(metaclass=ABCMeta):

    def __init__(self) -> None:
        self.__handlers: list[Handler] = []

    def add_handler(self, handler: Handler) -> None:
        self.__handlers.append(handler)
    
    @property
    def handlers(self) -> list[Handler]:
        return self.__handlers

    @abstractmethod
    def get_run_uri(self, id: str) -> str:
        """"""

    @abstractmethod
    def get_process_uri(self, id: str) -> str:
        """"""

    @abstractmethod
    def get_operation_uri(self, id: str) -> str:
        """"""

    # @abstractmethod
    # def create_run(self, metadata) -> str:
    #     """"""

    # @abstractmethod
    # def create_process(self, metadata) -> str:
    #     """"""

    # @abstractmethod
    # def create_operation(self, metadata) -> str:
    #     """"""

    # @abstractmethod
    # def update_run(self, id: str, metadata) -> None:
    #     """"""
        
    # @abstractmethod
    # def update_process(self, id: str, metadata) -> None:
    #     """"""

    # @abstractmethod
    # def update_operation(self, id: str, metadata) -> None:
    #     """"""

class Store(Store_):

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.__root = path or (Path.cwd() / 'experiments')
    
    def __create(self, prefix: str, *, id: str | None = None) -> str:
        id = id or str(uuid.uuid4())
        path = self.__get_path(prefix, id)
        assert not path.is_dir()
        # path.mkdir(parents=True)
        return id
    
    def __get_path(self, prefix: str, id: str) -> Path:
        path = self.__root / f'{prefix}/{id}'
        return path

    def get_run_uri(self, id: str) -> str:
        return self.__get_path('runs', id).as_uri()

    def get_process_uri(self, id: str) -> str:
        return self.__get_path('processes', id).as_uri()

    def get_operation_uri(self, id: str) -> str:
        return self.__get_path('operations', id).as_uri()
    
    def create_run(self, metadata: dict | None) -> str:
        run_id = self.__create('runs')
        storage_address = self.get_run_uri(run_id)
        [handler.create_run(run_id, storage_address, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]
        return run_id
    
    def finish_run(self, run_id: str, status: str = 'completed', metadata: dict | None = None) -> None:
        [handler.finish_run(run_id, status, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]

    def create_process(self, metadata: dict | None) -> str:
        process_id = self.__create('processes')
        storage_address = self.get_process_uri(process_id)
        [handler.create_process(process_id, storage_address, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]
        return process_id
    
    def finish_process(self, process_id: str, status: str = 'completed', metadata: dict | None = None) -> None:
        [handler.finish_process(process_id, status, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]

    def create_operation(self, metadata: dict | None) -> str:
        operation_id = self.__create('operations')
        storage_address = self.get_operation_uri(operation_id)
        [handler.create_operation(operation_id, storage_address, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]
        return operation_id

    def finish_operation(self, operation_id: str, status: str = 'completed', metadata: dict | None = None) -> None:
        [handler.finish_operation(operation_id, status, metadata) for handler in self.handlers]  # type: ignore[func-returns-value]

    def log_operation_text(self, operation_id: str, text: str, artifact_file: str) -> None:
        [handler.log_operation_text(operation_id, text, artifact_file) for handler in self.handlers]  # type: ignore[func-returns-value]

TokenTuple = tuple[PortAddress, dict[str, Any]]
@dataclasses.dataclass
class Job:
    operation: EntityDescription
    inputs: list[TokenTuple]
    outputs: list[TokenTuple] | None
    metadata: dict

class Run:

    def __init__(self, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        self.__metadata = metadata
        self.__running_jobs: dict[str, Job] = {}
        self.__complete_jobs: list[Job] = []

    def new_job(self, job_id: str, operation: EntityDescription, inputs: list[TokenTuple], metadata: dict | None = None) -> str:
        metadata = metadata or {}
        # job_id = str(uuid.uuid4())
        self.__running_jobs[job_id] = Job(operation, inputs, None, metadata)
        return job_id

    def complete_job(self, job_id: str, outputs: list[TokenTuple], metadata: dict | None = None) -> None:
        metadata = metadata or {}
        assert job_id in self.__running_jobs, job_id
        job = self.__running_jobs.pop(job_id)
        self.__complete_jobs.append(Job(job.operation, job.inputs, outputs, dict(job.metadata, **metadata)))

    @property
    def input(self) -> dict[str, Any]:
        assert len(self.__complete_jobs) > 0
        job = self.__complete_jobs[0]
        assert job.outputs is not None
        assert job.operation.id == "input"
        return {token[0].port_id: token[1] for token in job.outputs}

    @property
    def output(self) -> dict[str, Any]:
        assert len(self.__complete_jobs) > 1
        job = self.__complete_jobs[-1]
        assert job.operation.id == "output"
        return {token[0].port_id: token[1] for token in job.inputs}

    def jobs(self) -> list[Job]:
        return self.__complete_jobs  #XXX: copy?

    def running(self) -> ValuesView[Job]:
        return self.__running_jobs.values()

class RunHandler(Handler):

    def __init__(self) -> None:
        self.__run: Run | None = None

    @property    
    def run(self) -> Run:
        assert self.__run is not None
        return self.__run

    def create_run(self, run_id: str, storage_address: str, metadata: dict | None) -> None:
        self.__run = Run(metadata={'id': run_id, 'storage_address': storage_address})

    def create_process(self, process_id: str, storage_address: str, metadata: dict | None) -> None:
        assert self.__run is not None
        assert metadata is not None and all(key in metadata for key in ('id', 'base', 'inputs'))
        self.__run.new_job(process_id, EntityDescription(metadata['id'], metadata['base']), [dataclasses.astuple(token) for token in metadata['inputs']])

    def finish_process(self, process_id: str, status: str, metadata: dict | None) -> None:
        assert self.__run is not None
        assert metadata is not None and 'outputs' in metadata
        self.__run.complete_job(process_id, [dataclasses.astuple(token) for token in metadata['outputs']])

    def create_operation(self, operation_id: str, storage_address: str, metadata: dict | None) -> None:
        assert self.__run is not None

class Tracking:

    def __init__(self, url: str) -> None:
        self.__url = url

        self.__run_id: int | None = None
        
        self.__process_id_map: dict[str, int] = {}
        self.__operation_id_map: dict[str, int] = {}
        self.__process_dependencies: dict[str, list[str]] = {}
        self.__process_operation_map: dict[str, list[str]] = defaultdict(list)
    
    def create_run(self, project_id: int, user_id: int, checksum: str, file_name: str = 'dummy') -> int:
        response = requests.post(
            url=f'{self.__url}/runs/',
            data={
                "project_id": project_id,
                "file_name": file_name,
                "checksum": checksum,
                "user_id": user_id,
                "storage_address": '',  # not yet
            },
            verify=False
        )
        db_id = response.json()["id"]
        self.__run_id = db_id
        return db_id

    def start_run(self, storage_address: str = '', checksum: str = '') -> None:
        requests.patch(url=f'{self.__url}/runs/{self.__run_id}', data={"attribute": "storage_address", "new_value": storage_address}, verify=False)
        requests.patch(url=f'{self.__url}/runs/{self.__run_id}', data={"attribute": "checksum", "new_value": checksum}, verify=False)

        run_start_time = datetime.now().isoformat()
        requests.patch(url=f'{self.__url}/runs/{self.__run_id}', data={"attribute": "started_at", "new_value": run_start_time}, verify=False)
        requests.patch(url=f'{self.__url}/runs/{self.__run_id}', data={"attribute": "status", "new_value": "running"}, verify=False)

    def finish_run(self, status: str = 'completed') -> None:
        if self.__run_id is None:
            return
        run_finished_time = datetime.now().isoformat()
        requests.patch(url=f'{self.__url}/runs/{self.__run_id}', data={"attribute": "finished_at", "new_value": run_finished_time}, verify=False)
        requests.patch(url=f'{self.__url}/runs/{self.__run_id}', data={"attribute": "status", "new_value": status}, verify=False)

        self.__run_id = None
        self.__process_id_map = {}
        self.__operation_id_map = {}
        self.__process_dependencies = {}
        self.__process_operation_map = defaultdict(list)

    def start_process(self, process_id: str, name: str, storage_address: str = '', dependencies: list[str] | None = None) -> int:
        assert self.__run_id is not None
        response = requests.post(
            url=f'{self.__url}/processes/',
            data={
                "name": name,
                "run_id": self.__run_id,
                "storage_address": storage_address,
            },
            verify=False
        )
        db_id = response.json()["id"]
        self.__process_id_map[process_id] = db_id
        self.__process_dependencies[process_id] = copy.copy(dependencies) if dependencies is not None else []
        return db_id

    def finish_process(self, process_id: str, status: str = 'completed') -> None:
        pass

    def start_operation(self, operation_id: str, process_id: str, name: str, storage_address: str = '') -> int:
        assert self.__run_id is not None
        # src_operation_ids = [self.__last_operation_id] if self.__last_operation_id is not None else None

        response = requests.post(
            url=f'{self.__url}/operations/',
            data={
                "process_id": self.__process_id_map[process_id],
                "name": name,
                "status": 'running',
                "storage_address": storage_address,
                "is_transport": False,
                "is_data": False,
            },
            verify=False
        )
        db_id = response.json()["id"]
        self.__operation_id_map[operation_id] = db_id
        self.__process_operation_map[process_id].append(operation_id)

        run_start_time = datetime.now().isoformat()
        requests.patch(url=f'{self.__url}/operations/{db_id}', data={"attribute": "started_at", "new_value": run_start_time}, verify=False)

        for src_id in self.__process_dependencies[process_id]:
            if src_id in self.__process_operation_map:
                response = requests.post(
                    url=f'{self.__url}/edges/',
                    data={
                        "run_id": self.__run_id,
                        "from_id": self.__operation_id_map[self.__process_operation_map[src_id][-1]],
                        "to_id": db_id,
                    },
                    verify=False
                )

        return db_id
    
    def finish_operation(self, operation_id: str, status: str = 'completed') -> None:
        run_finished_time = datetime.now().isoformat()
        db_id = self.__operation_id_map[operation_id]
        requests.patch(url=f'{self.__url}/operations/{db_id}', data={"attribute": "finished_at", "new_value": run_finished_time}, verify=False)
        requests.patch(url=f'{self.__url}/operations/{db_id}', data={"attribute": "status", "new_value": status}, verify=False)

        # self.__last_operation_id = operation_id

    def get_operation_log(self, operation_id: str) -> str:
        db_id = self.__operation_id_map[operation_id]
        response = requests.get(url=f'{self.__url}/operations/{db_id}', verify=False)
        return response.json()["log"]

    def set_operation_log(self, operation_id: str, text: str) -> None:
        db_id = self.__operation_id_map[operation_id]
        requests.patch(url=f'{self.__url}/operations/{db_id}', data={"attribute": "log", "new_value": text}, verify=False)

class TrackingHandler(Handler):

    def __init__(self, url: str) -> None:
        self.__tracking = Tracking(url)

    @property
    def tracking(self) -> 'Tracking':
        return self.__tracking

    def create_run(self, run_id: str, storage_address: str, metadata: dict | None) -> None:
        assert metadata is not None and 'checksum' in metadata
        self.__tracking.start_run(storage_address, metadata['checksum'])

    def finish_run(self, run_id: str, status: str, metadta: dict | None) -> None:
        self.__tracking.finish_run(status)

    def create_process(self, process_id: str, storage_address: str, metadata: dict | None) -> None:
        assert metadata is not None and 'id' in metadata
        name = metadata['id']
        self.__tracking.start_process(process_id, name, storage_address, dependencies=metadata.get('dependencies', None))

    def finish_process(self, process_id: str, status: str, metadata: dict | None) -> None:
        self.__tracking.finish_process(process_id, status)

    def create_operation(self, operation_id: str, storage_address: str, metadata: dict | None) -> None:
        assert metadata is not None and 'process_id' in metadata
        assert metadata is not None and 'name' in metadata
        self.__tracking.start_operation(operation_id, metadata['process_id'], metadata['name'], storage_address)

    def finish_operation(self, operation_id: str, status: str, metadata: dict | None) -> None:
        self.__tracking.finish_operation(operation_id, status)

    def log_operation_text(self, operation_id: str, text: str, artifact_file: str) -> None:
        if PurePath(artifact_file) != PurePath('./log.txt'):
            return
        self.__tracking.set_operation_log(operation_id, text)
