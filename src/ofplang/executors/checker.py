#!/usr/bin/python
# -*- coding: utf-8 -*-
from logging import getLogger

from ..base.model import Model
from ..base.executor import ExecutorBase
from ..base.protocol import EntityDescription

logger = getLogger(__name__)


class TypeCheckError(RuntimeError):
    pass

class TypeChecker(ExecutorBase):

    async def execute(self, model: 'Model', process: EntityDescription, inputs: dict) -> dict:
        definition = model.get_definition_by_name(model.get_by_id(process.id).type)
        for port in definition.get('input', []):
            if port['id'] not in inputs:
                if 'default' not in port:
                    raise TypeCheckError(f"No corresponding input [{port['id']}]")
                else:
                    continue
            value = inputs[port['id']]
            assert 'type' in value
            logger.warn(f"TypeChecker: Comparing [{value['type']}] with [{port['type']}] in [{process.id}]")
            if not model.is_acceptable(value['type'], port['type']):
                raise TypeError(f"Invalid type [{value['type']}] was given [{port['id']}]")
        outputs = {port['id']: {'value': None, 'type': port['type']} for port in definition.get('output', [])}
        return outputs