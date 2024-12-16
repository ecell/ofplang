# ofplang

```
$ python setup.py install
$ cd samples
$ python sample1.py
```

## For developers

```
$ poetry install
$ poetry run ruff check ofplang
$ poetry run mypy ofplang
$ cd samples
$ poetry run python sample1.py
$ poetry run ofplang run protocol1.yaml --definitions definitions.yaml --cli-input-yaml inputs1.yaml
```